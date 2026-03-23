#!/usr/bin/env python3
import argparse
import csv
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline gate metrics from Symbiote DB.")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--since", help="ISO timestamp (overrides --days)")
    parser.add_argument("--until", help="ISO timestamp (default: now)")
    parser.add_argument("--out-dir", default="reports", help="Output directory for CSVs")
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

    decision_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    total_decisions = 0
    verify_count = 0

    for row in gate_rows:
        total_decisions += 1
        decision = (row["decision"] or "").strip()
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if decision == "VERIFY":
            verify_count += 1
        reasons_raw = row["evidence_refs_json"] or "{}"
        try:
            reasons = json.loads(reasons_raw).get("reasons", [])
        except json.JSONDecodeError:
            reasons = []
        for reason in reasons or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    snapshots = conn.execute(
        "SELECT timestamp, subject_state_json FROM subject_snapshots "
        "WHERE datetime(timestamp) >= datetime(?) AND datetime(timestamp) <= datetime(?) "
        "ORDER BY datetime(timestamp) ASC",
        (since.isoformat(), until.isoformat()),
    ).fetchall()

    conn.close()

    outcomes_path = os.path.join(args.out_dir, "gate_outcomes.csv")
    reasons_path = os.path.join(args.out_dir, "gate_reasons.csv")
    trends_path = os.path.join(args.out_dir, "organism_trends.csv")
    summary_path = os.path.join(args.out_dir, "gate_summary.json")

    with open(outcomes_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision", "count", "percent"])
        for decision, count in sorted(decision_counts.items()):
            percent = (count / total_decisions * 100.0) if total_decisions else 0.0
            writer.writerow([decision, count, f"{percent:.2f}"])

    with open(reasons_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason", "count", "percent"])
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            percent = (count / total_decisions * 100.0) if total_decisions else 0.0
            writer.writerow([reason, count, f"{percent:.2f}"])

    with open(trends_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "social_alignment", "fatigue"])
        for row in snapshots:
            timestamp = row["timestamp"]
            raw = row["subject_state_json"] or "{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            organism = (payload.get("state") or {}).get("organism") or {}
            writer.writerow(
                [
                    timestamp,
                    organism.get("social_alignment"),
                    organism.get("fatigue"),
                ]
            )

    summary = {
        "window_start": iso(since),
        "window_end": iso(until),
        "total_decisions": total_decisions,
        "verify_rate": (verify_count / total_decisions) if total_decisions else 0.0,
        "decisions": decision_counts,
        "reasons": reason_counts,
        "outputs": {
            "gate_outcomes": outcomes_path,
            "gate_reasons": reasons_path,
            "organism_trends": trends_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {outcomes_path}")
    print(f"Wrote {reasons_path}")
    print(f"Wrote {trends_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
