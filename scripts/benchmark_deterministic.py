#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from urllib import request as urlrequest


DEFAULT_CASES = [
    {
        "id": "baseline_greeting",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize the benefits of structured logging in two sentences."},
        ],
    },
    {
        "id": "planning_short",
        "messages": [
            {"role": "system", "content": "You are a pragmatic software engineer."},
            {"role": "user", "content": "List three risks of skipping schema validation."},
        ],
    },
    {
        "id": "reasoning_check",
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "If a cache TTL is 30s and requests arrive every 10s, how often will a miss occur?"},
        ],
    },
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def post_json(url: str, payload: dict, api_key: str | None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urlrequest.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def run_case(
    case: dict,
    base_url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": case["messages"],
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "tool_choice": "none",
    }
    start = time.time()
    response = post_json(f"{base_url}/chat/completions", payload, api_key)
    duration_ms = int((time.time() - start) * 1000)
    content = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return {
        "case_id": case["id"],
        "duration_ms": duration_ms,
        "output_hash": sha256_text(content),
        "output_preview": content[:200],
    }


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    rank = int(round((len(values) - 1) * max(0.0, min(1.0, pct))))
    return values[rank]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    cleaned = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def load_system_logs(db_path: str, since: str, until: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, category, run_id, payload FROM system_logs WHERE timestamp >= ? AND timestamp <= ?",
        (since, until),
    ).fetchall()
    conn.close()
    logs = []
    for row in rows:
        payload_raw = row["payload"] or "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        logs.append(
            {
                "timestamp": row["timestamp"],
                "category": row["category"],
                "run_id": row["run_id"],
                "payload": payload,
            }
        )
    return logs


def summarize_system_metrics(logs: list[dict]) -> dict:
    prompt_primary_hashes: set[str] = set()
    prompt_memory_hashes: set[str] = set()
    monologue_freshness_ms: list[int] = []
    tool_unknown_count = 0
    contract_violation_count = 0
    tts_starts: dict[str, int] = {}
    tts_ends: dict[str, int] = {}
    prompt_outcomes: dict[str, dict] = {}

    for entry in logs:
        payload = entry.get("payload") or {}
        event = payload.get("event")
        if event == "prompt_metrics":
            primary = payload.get("primary_prompt_hash")
            memory = payload.get("memory_prompt_hash")
            if primary:
                prompt_primary_hashes.add(primary)
            if memory:
                prompt_memory_hashes.add(memory)
        elif event == "monologue_freshness_ms":
            freshness = payload.get("freshness_ms")
            if isinstance(freshness, (int, float)):
                monologue_freshness_ms.append(int(freshness))
        elif event == "tool_unknown_name":
            tool_unknown_count += 1
        elif event == "tool_candidate_rejected" and payload.get("reason") == "UNKNOWN_TOOL":
            tool_unknown_count += 1
        elif event == "contract_violation":
            contract_violation_count += 1
        elif event == "tts_speak_start":
            run_id = payload.get("run_id") or entry.get("run_id") or "unknown"
            tts_starts[run_id] = tts_starts.get(run_id, 0) + 1
        elif event == "tts_speak_end":
            run_id = payload.get("run_id") or entry.get("run_id") or "unknown"
            tts_ends[run_id] = tts_ends.get(run_id, 0) + 1
        elif event == "pending_prompt_surface_attempted":
            prompt_id = payload.get("prompt_id")
            if prompt_id:
                prompt_outcomes[prompt_id] = {
                    "outcome": payload.get("outcome"),
                    "reason": payload.get("reason"),
                    "age_seconds": payload.get("prompt_age_seconds"),
                }

    freshness_p95 = percentile(monologue_freshness_ms, 0.95) if monologue_freshness_ms else 0
    tts_incomplete_runs = []
    for run_id, starts in tts_starts.items():
        ends = tts_ends.get(run_id, 0)
        if ends < starts:
            tts_incomplete_runs.append(run_id)

    pending_prompt_holds = 0
    for outcome in prompt_outcomes.values():
        if outcome.get("outcome") == "held":
            pending_prompt_holds += 1

    return {
        "prompt_primary_hashes": sorted(prompt_primary_hashes),
        "prompt_memory_hashes": sorted(prompt_memory_hashes),
        "monologue_freshness_p95_ms": freshness_p95,
        "tool_unknown_count": tool_unknown_count,
        "contract_violation_count": contract_violation_count,
        "tts_incomplete_runs": tts_incomplete_runs,
        "pending_prompt_holds": pending_prompt_holds,
        "pending_prompt_outcomes": prompt_outcomes,
    }


def evaluate_system_metrics(metrics: dict, max_freshness_ms: int) -> list[str]:
    failures: list[str] = []
    if metrics.get("monologue_freshness_p95_ms", 0) > max_freshness_ms:
        failures.append("monologue_freshness_p95_exceeded")
    if len(metrics.get("prompt_primary_hashes", [])) > 1:
        failures.append("prompt_primary_hash_variance")
    if len(metrics.get("prompt_memory_hashes", [])) > 1:
        failures.append("prompt_memory_hash_variance")
    if metrics.get("tool_unknown_count", 0) > 0:
        failures.append("unknown_tool_detected")
    if metrics.get("contract_violation_count", 0) > 0:
        failures.append("contract_violation_detected")
    if metrics.get("tts_incomplete_runs"):
        failures.append("tts_incomplete_runs")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic benchmark harness for Symbiote.")
    parser.add_argument("--base-url", default=os.getenv("SYMBIOTE_API_BASE", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("SYMBIOTE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("SYMBIOTE_MODEL", "default"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--baseline", default="benchmark_baseline.json")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--db-path", default=os.getenv("SYMBIOTE_DB_PATH"))
    parser.add_argument("--system-output", default="system_metrics.json")
    parser.add_argument("--max-monologue-freshness-ms", type=int, default=60000)
    args = parser.parse_args()

    start_ts = iso_now()
    results = []
    for _ in range(args.runs):
        for case in DEFAULT_CASES:
            results.append(
                run_case(
                    case,
                    args.base_url,
                    args.api_key,
                    args.model,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                )
            )
    end_ts = iso_now()

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"runs": results}, handle, indent=2)

    durations = [item["duration_ms"] for item in results if item.get("duration_ms") is not None]
    p50 = percentile(durations, 0.50)
    p95 = percentile(durations, 0.95)

    unique_hashes = {}
    for item in results:
        unique_hashes.setdefault(item["case_id"], set()).add(item["output_hash"])

    print("Benchmark complete.")
    for case_id, hashes in unique_hashes.items():
        status = "stable" if len(hashes) == 1 else "variance"
        print(f"{case_id}: {status} ({len(hashes)} unique hashes)")

    if args.write_baseline:
        with open(args.baseline, "w", encoding="utf-8") as handle:
            json.dump({"p50_ms": p50, "p95_ms": p95}, handle, indent=2)
        print(f"Wrote baseline to {args.baseline}")
    elif os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        baseline_p50 = baseline.get("p50_ms", 0)
        baseline_p95 = baseline.get("p95_ms", 0)
        if baseline_p50 and p50 > baseline_p50 * 1.10:
            print(f"P50 regression detected: {p50}ms > {baseline_p50}ms baseline")
            return 1
        if baseline_p95 and p95 > baseline_p95 * 1.10:
            print(f"P95 regression detected: {p95}ms > {baseline_p95}ms baseline")
            return 1

    if args.db_path:
        logs = load_system_logs(args.db_path, start_ts, end_ts)
        metrics = summarize_system_metrics(logs)
        with open(args.system_output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        failures = evaluate_system_metrics(metrics, args.max_monologue_freshness_ms)
        if failures:
            print("System metric failures:", ", ".join(failures))
            return 1
        print("System metrics OK.")
    else:
        print("No --db-path provided; skipping system metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
