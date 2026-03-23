#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict


def default_db_path() -> str:
    env = os.getenv("SYMBIOTE_DB_PATH")
    if env:
        return env
    appdata = os.getenv("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "com.symbiote.app", "symbiote.db")


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def decision_report_metrics(conn: sqlite3.Connection, since: str, until: str) -> Dict[str, Any]:
    rows = conn.execute(
        "SELECT report_json FROM decision_reports WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since, until),
    ).fetchall()
    total = 0
    gate_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    blocked_sum = 0
    fallback_used = 0
    cannot_respond = 0

    for (report_json,) in rows:
        total += 1
        try:
            report = json.loads(report_json)
        except json.JSONDecodeError:
            continue
        gate = report.get("gate_decision") or "NONE"
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        action = report.get("selected_action") or "unknown"
        action_counts[action] = action_counts.get(action, 0) + 1
        blocked_sum += int(report.get("blocked_candidates_count") or 0)
        if report.get("fallback_used"):
            fallback_used += 1
        if report.get("cannot_respond"):
            cannot_respond += 1

    avg_blocked = (blocked_sum / total) if total else 0
    return {
        "total": total,
        "gate_decisions": gate_counts,
        "selected_actions": action_counts,
        "avg_blocked_candidates": avg_blocked,
        "fallback_used": fallback_used,
        "cannot_respond": cannot_respond,
    }


def evidence_metrics(conn: sqlite3.Connection, since: str, until: str) -> Dict[str, Any]:
    evidence_sources = conn.execute(
        "SELECT COUNT(*) FROM evidence_sources WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since, until),
    ).fetchone()[0]
    evidence_links = conn.execute(
        "SELECT COUNT(*) FROM evidence_links WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?)",
        (since, until),
    ).fetchone()[0]
    return {
        "sources": int(evidence_sources),
        "links": int(evidence_links),
    }


def compare(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key, val in curr.items():
        if isinstance(val, dict):
            diff[key] = compare(prev.get(key, {}), val)
        elif isinstance(val, (int, float)):
            diff[key] = val - float(prev.get(key, 0))
        else:
            diff[key] = val
    return diff


def window(args, prefix: str) -> tuple[datetime, datetime]:
    since = parse_time(getattr(args, f"{prefix}_since"))
    until = parse_time(getattr(args, f"{prefix}_until"))
    minutes = getattr(args, f"{prefix}_minutes")
    now = datetime.now(timezone.utc)
    if since is None and until is None:
        until = now
        since = now - timedelta(minutes=max(minutes, 1))
    elif since is None:
        since = until - timedelta(minutes=max(minutes, 1))
    elif until is None:
        until = since + timedelta(minutes=max(minutes, 1))
    if since > until:
        since, until = until, since
    return since, until


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare kernel pipeline outputs across windows.")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--legacy-minutes", type=int, default=60, help="Legacy window minutes")
    parser.add_argument("--legacy-since")
    parser.add_argument("--legacy-until")
    parser.add_argument("--phased-minutes", type=int, default=60, help="Phased window minutes")
    parser.add_argument("--phased-since")
    parser.add_argument("--phased-until")
    parser.add_argument("--out", default="reports/kernel_pipeline_regression.json")
    args = parser.parse_args()

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return 1

    legacy_since, legacy_until = window(args, "legacy")
    phased_since, phased_until = window(args, "phased")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    legacy = {
        "window": {"since": iso(legacy_since), "until": iso(legacy_until)},
        "decision_reports": decision_report_metrics(conn, legacy_since.isoformat(), legacy_until.isoformat()),
        "evidence": evidence_metrics(conn, legacy_since.isoformat(), legacy_until.isoformat()),
    }
    phased = {
        "window": {"since": iso(phased_since), "until": iso(phased_until)},
        "decision_reports": decision_report_metrics(conn, phased_since.isoformat(), phased_until.isoformat()),
        "evidence": evidence_metrics(conn, phased_since.isoformat(), phased_until.isoformat()),
    }
    conn.close()

    diff = compare(legacy, phased)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "legacy": legacy,
        "phased": phased,
        "diff": diff,
        "notes": [
            "Run the target scenario under legacy pipeline mode, then phased mode, then compare windows.",
            "Set SYMBIOTE_KERNEL_PIPELINE_MODE=legacy or phased before running the scenario.",
        ],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote regression report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
