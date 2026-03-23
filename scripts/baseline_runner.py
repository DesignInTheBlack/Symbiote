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


def count_table(conn: sqlite3.Connection, table: str, col: str, since: str, until: str, where: str | None = None) -> int:
    base = f"SELECT COUNT(*) FROM {table} WHERE datetime({col}) >= datetime(?) AND datetime({col}) <= datetime(?)"
    if where:
        base += f" AND ({where})"
    row = conn.execute(base, (since, until)).fetchone()
    return int(row[0]) if row else 0


def gate_counts(conn: sqlite3.Connection, since: str, until: str) -> dict:
    rows = conn.execute(
        "SELECT decision, COUNT(*) as count FROM gate_decisions "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "GROUP BY decision",
        (since, until),
    ).fetchall()
    counts = {"ALLOW": 0, "ALLOW_WITH_NOTICE": 0, "ALLOW_WITH_AUDIT": 0, "VERIFY": 0, "DEFER": 0, "DENY": 0}
    for decision, count in rows:
        if decision in counts:
            counts[decision] = int(count)
        else:
            counts[decision] = int(count)
    return counts


def tool_counts(conn: sqlite3.Connection, since: str, until: str) -> dict:
    rows = conn.execute(
        "SELECT status, COUNT(*) as count FROM tool_dispatches "
        "WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) <= datetime(?) "
        "GROUP BY status",
        (since, until),
    ).fetchall()
    counts = {"success": 0, "failed": 0, "pending": 0}
    for status, count in rows:
        if status in counts:
            counts[status] = int(count)
        else:
            counts[status] = int(count)
    return counts


def diff_numbers(prev: dict, curr: dict) -> dict:
    diff = {}
    for key, val in curr.items():
        if isinstance(val, dict):
            diff[key] = diff_numbers(prev.get(key, {}), val)
        elif isinstance(val, (int, float)):
            diff[key] = val - float(prev.get(key, 0))
        else:
            diff[key] = val
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description="Symbiote baseline runner and report.")
    parser.add_argument("--db-path", default=default_db_path(), help="Path to symbiote.db")
    parser.add_argument("--minutes", type=int, default=60, help="Lookback window in minutes")
    parser.add_argument("--since", help="ISO start time (overrides --minutes)")
    parser.add_argument("--until", help="ISO end time (default: now)")
    parser.add_argument("--out", help="Output path for baseline report JSON")
    parser.add_argument("--latest", default="reports/latest_baseline.json", help="Path to latest baseline JSON")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = parse_time(args.since) or (now - timedelta(minutes=max(args.minutes, 1)))
    until = parse_time(args.until) or now
    if since > until:
        since, until = until, since

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return 1

    os.makedirs("reports", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    gate = gate_counts(conn, since.isoformat(), until.isoformat())
    gate_total = sum(gate.values())
    gate_refusals = gate.get("DENY", 0) + gate.get("DEFER", 0) + gate.get("VERIFY", 0)

    tool = tool_counts(conn, since.isoformat(), until.isoformat())
    memory_writes = count_table(conn, "memory_write_ledger", "created_at", since.isoformat(), until.isoformat())
    run_count = count_table(conn, "runs", "started_at", since.isoformat(), until.isoformat())
    message_count = count_table(conn, "messages", "created_at", since.isoformat(), until.isoformat())

    conn.close()

    report = {
        "window": {"since": iso(since), "until": iso(until)},
        "gate": {"counts": gate, "total": gate_total, "refusals": gate_refusals},
        "memory": {"write_count": memory_writes},
        "tools": {"success": tool.get("success", 0), "failed": tool.get("failed", 0), "pending": tool.get("pending", 0)},
        "runs": {"count": run_count},
        "messages": {"count": message_count},
    }

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"reports/baseline_{timestamp}.json"
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
        diff = diff_numbers(prev, report)
        diff_path = out_path.replace(".json", "_diff.json")
        with open(diff_path, "w", encoding="utf-8") as handle:
            json.dump(diff, handle, indent=2)
        print(f"Wrote diff: {diff_path}")

    with open(args.latest, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote baseline: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
