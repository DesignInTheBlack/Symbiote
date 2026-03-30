#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone


def default_db_path() -> str:
    env = os.getenv("SYMBIOTE_DB_PATH")
    if env:
        return env
    appdata = os.getenv("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "com.symbiote.app", "symbiote.db")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def compute_gate_penalty_metrics(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "gate_decisions"):
        return {"ok": False, "reason": "gate_decisions table missing"}
    rows = conn.execute(
        "SELECT metrics_json FROM gate_decisions ORDER BY datetime(created_at) DESC LIMIT 500"
    ).fetchall()
    total = 0
    with_penalty = 0
    penalty_sum = 0.0
    for row in rows:
        try:
            metrics = json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            continue
        total += 1
        penalty = metrics.get("gate_penalty")
        if isinstance(penalty, (int, float)) and penalty > 0:
            with_penalty += 1
            penalty_sum += float(penalty)
    avg_penalty = penalty_sum / with_penalty if with_penalty > 0 else 0.0
    return {
        "ok": True,
        "total_records": total,
        "penalty_count": with_penalty,
        "penalty_rate": (with_penalty / total) if total > 0 else 0.0,
        "avg_penalty": avg_penalty,
    }


def compute_plan_adherence(conn: sqlite3.Connection, since: datetime, until: datetime) -> dict:
    if not table_exists(conn, "tool_dispatches"):
        return {"ok": False, "reason": "tool_dispatches table missing"}
    total = conn.execute(
        "SELECT COUNT(*) FROM tool_dispatches WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since.isoformat(), until.isoformat()),
    ).fetchone()[0]
    with_plan = conn.execute(
        """
        SELECT COUNT(*) FROM tool_dispatches
        WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)
          AND plan_step_id IS NOT NULL
          AND trim(plan_step_id) != ''
        """,
        (since.isoformat(), until.isoformat()),
    ).fetchone()[0]
    return {
        "ok": True,
        "total_dispatches": total,
        "planned_dispatches": with_plan,
        "plan_adherence_rate": (with_plan / total) if total > 0 else 0.0,
    }


def compute_evidence_coverage(conn: sqlite3.Connection, since: datetime, until: datetime) -> dict:
    if not table_exists(conn, "ics_evidence_events"):
        return {"ok": False, "reason": "ics_evidence_events table missing"}
    rows = conn.execute(
        "SELECT source_type, COUNT(*) as count "
        "FROM ics_evidence_events "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "GROUP BY source_type",
        (since.isoformat(), until.isoformat()),
    ).fetchall()
    counts = {row["source_type"]: row["count"] for row in rows if row["source_type"]}
    required = [
        "candidate_created",
        "candidate_accepted",
        "arbitration_outcome",
        "gate_decision",
        "tool_dispatch",
        "tool_result",
        "tool_result_error",
        "memory_write",
        "memory_write_blocked",
        "self_report_snapshot",
    ]
    missing = [key for key in required if counts.get(key, 0) == 0]
    coverage = (len(required) - len(missing)) / len(required) if required else 1.0
    return {
        "ok": True,
        "coverage": coverage,
        "missing": missing,
        "counts": counts,
    }


def compute_outcome_quality(conn: sqlite3.Connection) -> dict:
    if not table_exists(conn, "self_model_controller_snapshots"):
        return {"ok": False, "reason": "self_model_controller_snapshots table missing"}
    rows = conn.execute(
        "SELECT state_json FROM self_model_controller_snapshots ORDER BY datetime(created_at) DESC LIMIT 50"
    ).fetchall()
    values = []
    for row in rows:
        try:
            state = json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            continue
        outcome_quality = state.get("outcome_quality")
        if isinstance(outcome_quality, (int, float)):
            values.append(float(outcome_quality))
    if not values:
        return {"ok": True, "samples": 0}
    avg_quality = sum(values) / len(values)
    return {"ok": True, "samples": len(values), "avg_outcome_quality": avg_quality}


def compute_evidence_strength(conn: sqlite3.Connection, since: datetime, until: datetime) -> dict:
    if not table_exists(conn, "ics_evidence_events"):
        return {"ok": False, "reason": "ics_evidence_events table missing"}
    row = conn.execute(
        "SELECT AVG(strength) as avg_strength, COUNT(*) as count "
        "FROM ics_evidence_events "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since.isoformat(), until.isoformat()),
    ).fetchone()
    avg_strength = row["avg_strength"] if row and row["avg_strength"] is not None else 0.0
    count = row["count"] if row and row["count"] is not None else 0
    return {"ok": True, "avg_strength": avg_strength, "count": count}


def group_counts(rows, key):
    counts = {}
    total = 0
    for row in rows:
        k = row.get(key) if isinstance(row, dict) else row[key]
        k = (k or "").strip()
        counts[k] = counts.get(k, 0) + 1
        total += 1
    return counts, total


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline audit for Evolve.md")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--since", help="ISO timestamp (overrides --days)")
    parser.add_argument("--until", help="ISO timestamp (default: now)")
    parser.add_argument("--out-dir", default="reports", help="Output directory")
    parser.add_argument("--sample-limit", type=int, default=5, help="Sample size for traces")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = parse_time(args.since) or (now - timedelta(days=max(args.days, 1)))
    until = parse_time(args.until) or now
    if since > until:
        since, until = until, since

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return 1

    ensure_out_dir(args.out_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    gate_rows = conn.execute(
        "SELECT decision, evidence_refs_json, created_at FROM gate_decisions "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since.isoformat(), until.isoformat()),
    ).fetchall()

    gate_decisions = {}
    gate_reasons = {}
    for row in gate_rows:
        decision = (row["decision"] or "").strip()
        gate_decisions[decision] = gate_decisions.get(decision, 0) + 1
        reasons_raw = row["evidence_refs_json"] or "{}"
        try:
            reasons = json.loads(reasons_raw).get("reasons", [])
        except json.JSONDecodeError:
            reasons = []
        for reason in reasons or []:
            gate_reasons[reason] = gate_reasons.get(reason, 0) + 1

    decision_reports = conn.execute(
        "SELECT report_json, created_at FROM decision_reports "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since.isoformat(), until.isoformat()),
    ).fetchall()

    selected_action_counts = {}
    selected_source_counts = {}
    accepted_kind_counts = {}
    for row in decision_reports:
        raw = row["report_json"] or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        action = (payload.get("selected_action") or "").strip()
        if action:
            selected_action_counts[action] = selected_action_counts.get(action, 0) + 1
        source = (payload.get("selected_action_source") or "").strip()
        if source:
            selected_source_counts[source] = selected_source_counts.get(source, 0) + 1
        candidate_scores = payload.get("candidate_scores", []) or []
        for candidate in candidate_scores:
            kind = (candidate.get("kind") or "").strip()
            if kind:
                accepted_kind_counts[kind] = accepted_kind_counts.get(kind, 0) + 1

    evidence_rows = conn.execute(
        "SELECT source_type, COUNT(*) as count "
        "FROM ics_evidence_events "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "GROUP BY source_type",
        (since.isoformat(), until.isoformat()),
    ).fetchall()
    evidence_counts = {row["source_type"]: row["count"] for row in evidence_rows}

    evidence_by_day = conn.execute(
        "SELECT strftime('%Y-%m-%d', created_at) as day, source_type, COUNT(*) as count "
        "FROM ics_evidence_events "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "GROUP BY day, source_type ORDER BY day ASC",
        (since.isoformat(), until.isoformat()),
    ).fetchall()
    evidence_day_map = {}
    for row in evidence_by_day:
        day = row["day"]
        source_type = row["source_type"]
        count = row["count"]
        bucket = evidence_day_map.setdefault(day, {})
        bucket[source_type] = count

    sample_limit = max(1, args.sample_limit)
    sample_decisions = conn.execute(
        "SELECT report_json, created_at FROM decision_reports "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "ORDER BY datetime(created_at) DESC LIMIT ?",
        (since.isoformat(), until.isoformat(), sample_limit),
    ).fetchall()

    sample_tools = conn.execute(
        "SELECT action_id, tool_name, status, failure_kind, created_at, updated_at, result_text "
        "FROM tool_dispatches "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "ORDER BY datetime(created_at) DESC LIMIT ?",
        (since.isoformat(), until.isoformat(), sample_limit),
    ).fetchall()

    sample_memory = conn.execute(
        "SELECT id, event_type, summary, timestamp, run_id FROM episodic_events "
        "WHERE datetime(timestamp) >= datetime(?) AND datetime(timestamp) <= datetime(?) "
        "ORDER BY datetime(timestamp) DESC LIMIT ?",
        (since.isoformat(), until.isoformat(), sample_limit),
    ).fetchall()

    report = {
        "window_start": iso(since),
        "window_end": iso(until),
        "gate_decisions": gate_decisions,
        "gate_reasons": gate_reasons,
        "arbitration_selected_actions": selected_action_counts,
        "arbitration_selected_sources": selected_source_counts,
        "accepted_candidate_kinds": accepted_kind_counts,
        "evidence_counts": evidence_counts,
        "evidence_by_day": evidence_day_map,
        "baseline_metrics": {
            "gate_penalty": compute_gate_penalty_metrics(conn),
            "plan_adherence": compute_plan_adherence(conn, since, until),
            "evidence_coverage": compute_evidence_coverage(conn, since, until),
            "outcome_quality": compute_outcome_quality(conn),
            "evidence_strength": compute_evidence_strength(conn, since, until),
        },
        "samples": {
            "decision_reports": [
                {
                    "created_at": row["created_at"],
                    "report_json": row["report_json"],
                }
                for row in sample_decisions
            ],
            "tool_dispatches": [dict(row) for row in sample_tools],
            "episodic_events": [dict(row) for row in sample_memory],
        },
    }

    out_path = os.path.join(args.out_dir, "evolve_baseline_report.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote {out_path}")

    baseline_dir = os.path.join("artifacts")
    ensure_out_dir(baseline_dir)
    baseline_path = os.path.join(baseline_dir, "evolve_baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {baseline_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
