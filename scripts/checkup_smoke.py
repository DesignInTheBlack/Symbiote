import argparse
import json
import os
import sqlite3
import sys

DEFAULT_DB = r"C:\\Users\\desig\\AppData\\Roaming\\com.symbiote.app\\symbiote.db"


def looks_jsonish(text: str) -> bool:
    trimmed = (text or "").strip()
    if len(trimmed) < 2:
        return False
    wrapped = (trimmed.startswith("{") and trimmed.endswith("}")) or (
        trimmed.startswith("[") and trimmed.endswith("]")
    )
    if not wrapped:
        return False
    lower = trimmed.lower()
    if any(key in lower for key in ["\"stance\"", "\"candidates\"", "\"done\"", "\"message\""]):
        return True
    try:
        json.loads(trimmed)
        return True
    except Exception:
        return False


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Symbiote checkup smoke test")
    parser.add_argument("--db", default=os.environ.get("SYMBIOTE_DB", DEFAULT_DB))
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    failures = 0

    if not table_exists(conn, "deferred_emits"):
        print("FAIL: deferred_emits table missing")
        failures += 1
    else:
        print("OK: deferred_emits table present")

    if table_exists(conn, "messages"):
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM messages
            WHERE role = 'assistant'
              AND status IN ('cancelled','error')
              AND (
                metadata IS NULL
                OR json_valid(metadata) = 0
                OR json_extract(metadata, '$.surface') IS NULL
                OR json_extract(metadata, '$.surface') != 0
              )
            """
        ).fetchone()
        count = row["c"] if row else 0
        if count > 0:
            print(f"FAIL: {count} cancelled/error assistant messages are still surfaceable")
            failures += 1
        else:
            print("OK: cancelled/error assistant messages are suppressed")

    if table_exists(conn, "pending_user_prompts"):
        rows = conn.execute(
            "SELECT id, prompt FROM pending_user_prompts WHERE prompt LIKE '{%' OR prompt LIKE '[%'")
        jsonish = [row["id"] for row in rows if looks_jsonish(row["prompt"])]
        if jsonish:
            print(f"FAIL: JSON-like pending prompts detected: {', '.join(jsonish)}")
            failures += 1
        else:
            print("OK: no JSON-like pending prompts")

    if table_exists(conn, "system_health_snapshots"):
        row = conn.execute(
            "SELECT metrics_json FROM system_health_snapshots ORDER BY datetime(timestamp) DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                metrics = json.loads(row[0])
                combined = metrics.get("scorecard", {}).get("combined_score")
                if combined is None:
                    print("OK: combined_score is null when outcomes missing")
                else:
                    print("WARN: combined_score is not null (verify outcomes present)")
            except Exception as exc:
                print(f"WARN: failed to parse system_health_snapshots metrics: {exc}")
        else:
            print("WARN: no system_health_snapshots rows found")

    conn.close()

    if failures:
        print(f"Smoke test failed with {failures} issue(s).")
        return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
