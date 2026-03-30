import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta


DEFAULT_DB = r"C:\\Users\\desig\\AppData\\Roaming\\com.symbiote.app\\symbiote.db"
DEFAULT_BASELINE = os.path.join("artifacts", "evolve_baseline.json")


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def parse_json(value: str):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def load_baseline(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def delta_percent(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline


def gate_penalty_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "gate_decisions"):
        return {"ok": False, "reason": "gate_decisions table missing"}
    rows = conn.execute(
        "SELECT metrics_json FROM gate_decisions ORDER BY datetime(created_at) DESC LIMIT 500"
    ).fetchall()
    total = 0
    with_penalty = 0
    penalty_sum = 0.0
    penalty_reasons = {}
    for row in rows:
        metrics = parse_json(row[0])
        if not isinstance(metrics, dict):
            continue
        total += 1
        penalty = metrics.get("gate_penalty")
        if isinstance(penalty, (int, float)) and penalty > 0:
            with_penalty += 1
            penalty_sum += float(penalty)
        reasons = metrics.get("penalty_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str):
                    penalty_reasons[reason] = penalty_reasons.get(reason, 0) + 1
    avg_penalty = penalty_sum / with_penalty if with_penalty > 0 else 0.0
    top_reasons = sorted(penalty_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "ok": True,
        "total_records": total,
        "penalty_count": with_penalty,
        "penalty_rate": (with_penalty / total) if total > 0 else 0.0,
        "avg_penalty": avg_penalty,
        "top_reasons": top_reasons,
    }


def evidence_coverage_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "ics_evidence_events"):
        return {"ok": False, "reason": "ics_evidence_events table missing"}
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    rows = conn.execute(
        """
        SELECT source_type, COUNT(*) as count
        FROM ics_evidence_events
        WHERE datetime(created_at) >= datetime(?)
        GROUP BY source_type
        """,
        (since,),
    ).fetchall()
    counts = {row[0]: row[1] for row in rows if row[0]}
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


def plan_adherence_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "tool_dispatches"):
        return {"ok": False, "reason": "tool_dispatches table missing"}
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    total = conn.execute(
        "SELECT COUNT(*) FROM tool_dispatches WHERE datetime(created_at) >= datetime(?)",
        (since,),
    ).fetchone()[0]
    with_plan = conn.execute(
        """
        SELECT COUNT(*) FROM tool_dispatches
        WHERE datetime(created_at) >= datetime(?)
          AND plan_step_id IS NOT NULL
          AND trim(plan_step_id) != ''
        """,
        (since,),
    ).fetchone()[0]
    return {
        "ok": True,
        "total_dispatches": total,
        "planned_dispatches": with_plan,
        "plan_adherence_rate": (with_plan / total) if total > 0 else 0.0,
    }


def confidence_drift_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "self_model_controller_snapshots"):
        return {"ok": False, "reason": "self_model_controller_snapshots table missing"}
    rows = conn.execute(
        "SELECT state_json FROM self_model_controller_snapshots ORDER BY datetime(created_at) DESC LIMIT 50"
    ).fetchall()
    if not rows:
        return {"ok": True, "samples": 0}
    deltas = []
    for row in rows:
        state = parse_json(row[0])
        if not isinstance(state, dict):
            continue
        confidence = state.get("confidence")
        outcome_quality = state.get("outcome_quality")
        if isinstance(confidence, (int, float)) and isinstance(outcome_quality, (int, float)):
            deltas.append(abs(float(confidence) - float(outcome_quality)))
    if not deltas:
        return {"ok": True, "samples": 0}
    avg_delta = sum(deltas) / len(deltas)
    return {
        "ok": True,
        "samples": len(deltas),
        "avg_abs_delta": avg_delta,
        "max_abs_delta": max(deltas),
    }


def outcome_quality_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "self_model_controller_snapshots"):
        return {"ok": False, "reason": "self_model_controller_snapshots table missing"}
    rows = conn.execute(
        "SELECT state_json FROM self_model_controller_snapshots ORDER BY datetime(created_at) DESC LIMIT 50"
    ).fetchall()
    values = []
    for row in rows:
        state = parse_json(row[0])
        if not isinstance(state, dict):
            continue
        quality = state.get("outcome_quality")
        if isinstance(quality, (int, float)):
            values.append(float(quality))
    if not values:
        return {"ok": True, "samples": 0}
    avg_quality = sum(values) / len(values)
    return {
        "ok": True,
        "samples": len(values),
        "avg_outcome_quality": avg_quality,
        "max_outcome_quality": max(values),
    }


def evidence_strength_metrics(conn: sqlite3.Connection):
    if not table_exists(conn, "ics_evidence_events"):
        return {"ok": False, "reason": "ics_evidence_events table missing"}
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    row = conn.execute(
        """
        SELECT AVG(strength) as avg_strength, COUNT(*) as count
        FROM ics_evidence_events
        WHERE datetime(created_at) >= datetime(?)
        """,
        (since,),
    ).fetchone()
    avg_strength = row[0] if row and row[0] is not None else 0.0
    count = row[1] if row and row[1] is not None else 0
    return {
        "ok": True,
        "avg_strength": avg_strength,
        "count": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolve metrics report")
    parser.add_argument("--db", default=os.environ.get("SYMBIOTE_DB", DEFAULT_DB))
    parser.add_argument("--baseline", default=os.environ.get("SYMBIOTE_EVOLVE_BASELINE", DEFAULT_BASELINE))
    parser.add_argument("--tripwire-threshold", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    baseline = load_baseline(args.baseline)
    baseline_metrics = (baseline or {}).get("baseline_metrics", {})
    baseline_timestamp = (baseline or {}).get("window_end")

    gate_metrics = gate_penalty_metrics(conn)
    evidence_metrics = evidence_coverage_metrics(conn)
    plan_metrics = plan_adherence_metrics(conn)
    confidence_metrics = confidence_drift_metrics(conn)
    outcome_metrics = outcome_quality_metrics(conn)
    strength_metrics = evidence_strength_metrics(conn)

    print("Gate penalty usage:")
    print(json.dumps(gate_metrics, indent=2))
    print()

    print("Evidence coverage (24h):")
    print(json.dumps(evidence_metrics, indent=2))
    print()

    print("Plan adherence (24h):")
    print(json.dumps(plan_metrics, indent=2))
    print()

    print("Confidence calibration drift:")
    print(json.dumps(confidence_metrics, indent=2))
    print()

    print("Outcome quality:")
    print(json.dumps(outcome_metrics, indent=2))
    print()

    print("Evidence strength (24h):")
    print(json.dumps(strength_metrics, indent=2))
    print()

    if baseline_metrics:
        deltas = []
        baseline_gate = baseline_metrics.get("gate_penalty", {})
        baseline_evidence = baseline_metrics.get("evidence_coverage", {})
        baseline_plan = baseline_metrics.get("plan_adherence", {})
        baseline_quality = baseline_metrics.get("outcome_quality", {})
        baseline_strength = baseline_metrics.get("evidence_strength", {})

        if gate_metrics.get("ok") and baseline_gate.get("ok"):
            deltas.append({
                "metric": "gate_penalty_rate",
                "current": gate_metrics.get("penalty_rate", 0.0),
                "baseline": baseline_gate.get("penalty_rate", 0.0),
            })
        if evidence_metrics.get("ok") and baseline_evidence.get("ok"):
            deltas.append({
                "metric": "evidence_coverage",
                "current": evidence_metrics.get("coverage", 0.0),
                "baseline": baseline_evidence.get("coverage", 0.0),
            })
        if plan_metrics.get("ok") and baseline_plan.get("ok"):
            deltas.append({
                "metric": "plan_adherence_rate",
                "current": plan_metrics.get("plan_adherence_rate", 0.0),
                "baseline": baseline_plan.get("plan_adherence_rate", 0.0),
            })
        if outcome_metrics.get("ok") and baseline_quality.get("ok"):
            deltas.append({
                "metric": "outcome_quality_avg",
                "current": outcome_metrics.get("avg_outcome_quality", 0.0),
                "baseline": baseline_quality.get("avg_outcome_quality", 0.0),
            })
        if strength_metrics.get("ok") and baseline_strength.get("ok"):
            deltas.append({
                "metric": "evidence_strength_avg",
                "current": strength_metrics.get("avg_strength", 0.0),
                "baseline": baseline_strength.get("avg_strength", 0.0),
            })

        tripwire = []
        for entry in deltas:
            delta = delta_percent(entry["current"], entry["baseline"])
            entry["delta_percent"] = delta
            if abs(delta) >= args.tripwire_threshold:
                tripwire.append(entry)

        print("Baseline comparison:")
        print(json.dumps({
            "baseline_timestamp": baseline_timestamp,
            "tripwire_threshold": args.tripwire_threshold,
            "deltas": deltas,
            "tripwire": tripwire,
        }, indent=2))
    else:
        print("Baseline comparison: none (baseline file not found)")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
