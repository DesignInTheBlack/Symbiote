#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta


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


def count_logs(conn: sqlite3.Connection, event: str, since: str, until: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM system_logs WHERE datetime(timestamp) >= datetime(?) AND datetime(timestamp) <= datetime(?) AND json_extract(payload, '$.event') = ?",
        (since, until, event),
    ).fetchone()
    return int(row[0]) if row else 0


def count_outcomes(conn: sqlite3.Connection, since: str, until: str) -> dict:
    rows = conn.execute(
        "SELECT verdict, COUNT(*) FROM outcome_events WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) GROUP BY verdict",
        (since, until),
    ).fetchall()
    counts = {"confirm": 0, "disconfirm": 0, "inconclusive": 0, "other": 0}
    for verdict, count in rows:
        key = verdict if verdict in counts else "other"
        counts[key] = int(count)
    total = sum(counts.values())
    denom = counts["confirm"] + counts["disconfirm"]
    accuracy = (counts["confirm"] / denom) if denom > 0 else 0.0
    return {
        "total": total,
        **counts,
        "accuracy": accuracy,
    }


def average_confidence(conn: sqlite3.Connection, table: str) -> float:
    row = conn.execute(
        f"SELECT AVG(confidence) FROM {table} WHERE status = 'active'",
    ).fetchone()
    if not row:
        return 0.0
    value = row[0]
    return float(value) if value is not None else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Symbiote cognition scorecard.")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--minutes", type=int, default=180, help="Lookback window in minutes")
    parser.add_argument("--since", help="ISO start time (overrides --minutes)")
    parser.add_argument("--until", help="ISO end time (default: now)")
    parser.add_argument("--out", help="Output path for scorecard JSON")
    parser.add_argument("--latest", default="reports/latest_scorecard.json", help="Path to latest scorecard JSON")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = parse_time(args.since) or (now - timedelta(minutes=max(args.minutes, 1)))
    until = parse_time(args.until) or now
    if since > until:
        since, until = until, since

    if not os.path.exists(args.db_path):
        print(f"DB not found: {args.db_path}")
        return 1

    os.makedirs("reports", exist_ok=True)

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row

    memory_drift = count_logs(conn, "memory_validation_drift", since.isoformat(), until.isoformat())
    memory_runs = count_logs(conn, "memory_validation_run", since.isoformat(), until.isoformat())
    telemetry_drift = count_logs(conn, "telemetry_calibration_drift", since.isoformat(), until.isoformat())
    telemetry_runs = count_logs(conn, "telemetry_calibration_run", since.isoformat(), until.isoformat())

    outcomes = count_outcomes(conn, since.isoformat(), until.isoformat())

    ics_conf = average_confidence(conn, "ics_beliefs")
    self_conf = average_confidence(conn, "self_beliefs")

    drift_penalty = (memory_drift + telemetry_drift) * 0.02
    combined_score = max(0.0, outcomes["accuracy"] - drift_penalty)

    conn.close()

    report = {
        "window": {"since": iso(since), "until": iso(until)},
        "memory": {
            "validation_runs": memory_runs,
            "drift_events": memory_drift,
            "avg_confidence": ics_conf,
        },
        "self_memory": {
            "avg_confidence": self_conf,
        },
        "telemetry": {
            "calibration_runs": telemetry_runs,
            "drift_events": telemetry_drift,
        },
        "outcomes": outcomes,
        "combined_score": combined_score,
        "notes": [
            "combined_score = outcome_accuracy - 0.02 * (memory_drift + telemetry_drift)",
            "Use combined_score for release gating alongside detailed metrics.",
        ],
    }

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"reports/cognition_scorecard_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    prev = None
    if os.path.exists(args.latest):
        try:
            with open(args.latest, "r", encoding="utf-8") as handle:
                prev = json.load(handle)
        except json.JSONDecodeError:
            prev = None

    if prev:
        diff = {
            "outcome_accuracy": report["outcomes"]["accuracy"] - prev.get("outcomes", {}).get("accuracy", 0),
            "memory_drift": report["memory"]["drift_events"] - prev.get("memory", {}).get("drift_events", 0),
            "telemetry_drift": report["telemetry"]["drift_events"] - prev.get("telemetry", {}).get("drift_events", 0),
            "combined_score": report["combined_score"] - prev.get("combined_score", 0),
        }
        diff_path = out_path.replace(".json", "_diff.json")
        with open(diff_path, "w", encoding="utf-8") as handle:
            json.dump(diff, handle, indent=2)
        print(f"Wrote diff: {diff_path}")

    with open(args.latest, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote scorecard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
