import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from collections import Counter

DEFAULT_DB = os.path.expandvars(r"C:\Users\desig\AppData\Roaming\com.symbiote.app\symbiote.db")


def parse_payload(raw):
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Unity diagnostic snapshot")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to symbiote.db")
    parser.add_argument("--window-mins", type=int, default=30, help="Window in minutes")
    parser.add_argument("--log-baseline", action="store_true", help="Insert unity_baseline event into system_logs")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=args.window_mins)
    window_start_iso = window_start.isoformat()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT type, payload FROM event_ledger WHERE timestamp >= ?",
        (window_start_iso,),
    )
    rows = cur.fetchall()

    cur.execute(
        "SELECT payload FROM system_logs WHERE timestamp >= ?",
        (window_start_iso,),
    )
    log_rows = cur.fetchall()
    log_payloads = [parse_payload(r["payload"]) for r in log_rows]

    pred_fail = [r for r in rows if r["type"] == "prediction_generation_failed"]
    pred_reasons = Counter(parse_payload(r["payload"]).get("reason") for r in pred_fail)

    monologue_counts = Counter(r["type"] for r in rows if r["type"].startswith("monologue_"))

    contract = [r for r in rows if r["type"] == "contract_violation"]
    contract_policy = Counter()
    contract_reason = Counter()
    for r in contract:
        p = parse_payload(r["payload"])
        contract_policy[p.get("policy_id")] += 1
        contract_reason[p.get("reason")] += 1

    mem_block = [r for r in rows if r["type"] == "memory_write_blocked"]
    mem_reason = Counter()
    mem_source = Counter()
    for r in mem_block:
        p = parse_payload(r["payload"])
        mem_reason[p.get("reason")] += 1
        mem_source[p.get("source")] += 1

    cur.execute("SELECT COUNT(*) as c FROM subject_snapshots WHERE timestamp >= ?", (window_start_iso,))
    snapshots = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) as c FROM gate_decisions WHERE created_at >= ?", (window_start_iso,))
    gate_decisions = cur.fetchone()["c"]
    cur.execute(
        """
        SELECT COUNT(*) as c
        FROM subject_snapshots s
        WHERE s.timestamp >= ?
          AND EXISTS (SELECT 1 FROM gate_decisions g WHERE g.snapshot_hash = s.snapshot_hash)
        """,
        (window_start_iso,),
    )
    snapshots_with_gate = cur.fetchone()["c"]

    prediction_repairs = [p for p in log_payloads if p.get("event") == "prediction_json_repair_used"]
    prediction_complete = [p for p in log_payloads if p.get("event") == "prediction_generation_complete"]
    prediction_repair_rate = (
        len(prediction_repairs) / len(prediction_complete)
        if len(prediction_complete) > 0
        else 0.0
    )
    residual_impacts = [
        p.get("impact_pct")
        for p in log_payloads
        if p.get("event") == "residual_shadow_impact"
        and isinstance(p.get("impact_pct"), (int, float))
    ]
    residual_shadow_impact_pct = (
        sum(residual_impacts) / len(residual_impacts)
        if residual_impacts
        else 0.0
    )
    residual_shadow_would_change = sum(
        1
        for p in log_payloads
        if p.get("event") == "residual_shadow_impact"
        and bool(p.get("would_change_winner"))
    )

    report = {
        "window_start": window_start_iso,
        "window_end": now.isoformat(),
        "prediction_generation_failed": {
            "total": len(pred_fail),
            "reasons": {k: v for k, v in pred_reasons.items() if k is not None},
        },
        "prediction_health": {
            "json_repair_rate": prediction_repair_rate,
            "json_repair_count": len(prediction_repairs),
            "residual_shadow_impact_pct": residual_shadow_impact_pct,
            "residual_shadow_would_change": residual_shadow_would_change,
        },
        "monologue": {
            "parse_failed": monologue_counts.get("monologue_parse_failed", 0),
            "tick_timeout": monologue_counts.get("monologue_tick_timeout", 0),
            "tick_retry": monologue_counts.get("monologue_tick_retry", 0),
        },
        "contract_violation": {
            "total": len(contract),
            "policy": {k: v for k, v in contract_policy.items() if k is not None},
            "reason": {k: v for k, v in contract_reason.items() if k is not None},
        },
        "memory_write_blocked": {
            "total": len(mem_block),
            "reason": {k: v for k, v in mem_reason.items() if k is not None},
            "source": {k: v for k, v in mem_source.items() if k is not None},
        },
        "snapshots": {
            "subject_snapshots": snapshots,
            "gate_decisions": gate_decisions,
            "snapshots_with_gate": snapshots_with_gate,
        },
    }

    print(json.dumps(report, indent=2))

    if args.log_baseline:
        payload = {
            "event": "unity_baseline",
            "window_start": report["window_start"],
            "window_end": report["window_end"],
            "counts": report,
        }
        cur.execute(
            "INSERT INTO system_logs (id, timestamp, level, category, run_id, trace_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                now.isoformat(),
                "info",
                "system",
                None,
                None,
                json.dumps(payload),
            ),
        )
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
