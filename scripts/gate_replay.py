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


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay gate decisions from logged inputs.")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--since", help="ISO timestamp (overrides --days)")
    parser.add_argument("--until", help="ISO timestamp (default: now)")
    parser.add_argument("--out", default="reports/gate_replay.csv", help="Output CSV path")
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, run_id, payload FROM system_logs "
        "WHERE json_extract(payload, '$.event') = 'gate_decision_inputs' "
        "AND datetime(timestamp) >= datetime(?) AND datetime(timestamp) <= datetime(?) "
        "ORDER BY datetime(timestamp) ASC",
        (since.isoformat(), until.isoformat()),
    ).fetchall()
    conn.close()

    total = 0
    matched = 0
    changed = 0

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "timestamp",
            "run_id",
            "candidate_id",
            "candidate_kind",
            "soft_decision",
            "legacy_decision",
            "enforced_decision",
            "expected_enforced",
            "match",
            "shadow_mode",
            "rollout_percent",
            "rollout_bucket",
        ])
        for row in rows:
            payload_raw = row["payload"] or "{}"
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            total += 1
            soft_decision = payload.get("soft_decision")
            legacy_decision = payload.get("legacy_decision")
            enforced = payload.get("enforced_decision")
            shadow_mode = bool(payload.get("shadow_mode", False))
            rollout_percent = int(payload.get("rollout_percent", 100))
            rollout_bucket = int(payload.get("rollout_bucket", 0))
            use_soft = (not shadow_mode) and (rollout_percent >= 100 or rollout_bucket < rollout_percent)
            expected = soft_decision if use_soft else legacy_decision
            is_match = expected == enforced
            matched += 1 if is_match else 0
            changed += 1 if soft_decision != legacy_decision else 0
            writer.writerow([
                row["timestamp"],
                row["run_id"],
                payload.get("candidate_id"),
                payload.get("candidate_kind"),
                soft_decision,
                legacy_decision,
                enforced,
                expected,
                "yes" if is_match else "no",
                shadow_mode,
                rollout_percent,
                rollout_bucket,
            ])

    print(f"Wrote {args.out}")
    if total:
        print(f"Matches: {matched}/{total} ({matched / total:.2%})")
        print(f"Soft vs legacy changed: {changed}/{total} ({changed / total:.2%})")
    else:
        print("No gate_decision_inputs events found in window.")
    print(f"Window: {iso(since)} to {iso(until)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
