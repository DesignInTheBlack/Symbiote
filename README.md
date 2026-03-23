# Symbiote

**Governed cognition for assistants that must be accountable.**

Symbiote is a local-first Tauri desktop app with a Rust kernel that mediates model output. The model proposes. The kernel decides. Tool calls, memory writes, and policy gates are enforced in code and logged.

If you want an assistant that shows its work instead of hiding it, Symbiote is built for that.

---

## The Pitch in One Minute

Most assistants optimize for fluency. Symbiote optimizes for accountability.

That single choice changes everything:
- The model is treated as a proposal engine, not the final authority.
- Memory is structured, validated, and traceable.
- The UI exposes traces, health, and memory graphs as first-class surfaces.

This is not a chatbot with a skin. It is a governance engine wrapped in a UI.

---

## What It Is

Symbiote is a **governed cognition engine** wrapped in a desktop application. It exists for long sessions, high-stakes use, and the need to explain why the system said or did something.

---

## What Makes It Different

### 1) A Kernel That Decides
Model output is not an answer. It is a candidate. The kernel checks evidence and policy before anything is committed. This introduces latency, but it creates a trail you can inspect and correct.

### 2) Memory That Can Be Audited
Memory is encoded in a DSL (ICS v4.1), not a raw chat log. Facts, relations, time, confidence, and provenance are explicit. Conflicts are recorded rather than silently overwritten.

Above the memory layer, a world model reconciles what the system currently believes to be true as new evidence arrives.

### 3) An Instrument Panel, Not a Shell
The UI is built for operators. Trace view, health panels, and the memory graph make the system legible. You do not have to trust it. You can see it.

---

## How It Works (System View)

1. Ingest user input.
2. Build a structured, budgeted prompt with policy, memory, and workspace context.
3. Generate candidate outputs from the model, not final answers.
4. Arbitrate and gate candidates against evidence and policy.
5. Commit accepted candidates to memory, summaries, tools, or user output.
6. Continue background cognition such as summaries, consolidation, and health checks.

---

## Model Roles and Responsibilities

Symbiote supports distinct models for different responsibilities. These are configured in Settings and can point at the same or different endpoints.

| Role | Setting | Used for |
| --- | --- | --- |
| Primary model | `active_model_id` | User-facing responses, tool-call proposals, and general candidate generation. Also the fallback for any task without a dedicated model. |
| Summary model | `summarization_model` and `summarization_api_url` | Rolling and live summaries, inner summaries, memory pass extraction, working-memory reflection, response rewrites, counterfactual simulation, and dream consolidation. Falls back to the primary model if not set. |
| Embedding model | `embedding_model` | Embedding-based memory retrieval and semantic search. Optional. |
| JSON compatibility list | `json_only_disabled_models` | Models that should not be forced into strict JSON-only responses when a request expects JSON. |

---

## Model Choice Recommendations

Choose models based on the job they do inside the system, not just raw benchmark scores.

1. Single-model baseline
Set `active_model_id` only. Symbiote uses the same model for primary responses and all background summarization. This is the simplest setup, but it is typically slower and more expensive.

2. Split primary and summary models (recommended)
Set `active_model_id` for the main assistant and `summarization_model` for summaries, memory passes, and background reflections. If you want a separate endpoint for summaries, set `summarization_api_url`. This keeps primary responses high quality while reducing cost and latency for maintenance tasks.

3. Add embeddings for stronger memory retrieval
Set `embedding_model` to enable embedding-based retrieval. This improves semantic recall and long-session memory quality.

Selection criteria:
- Primary model: strong reasoning, long-context support, and reliable structured output. JSON compliance matters because many kernel phases rely on machine-readable outputs.
- Summary model: fast, cost-efficient, and strict with JSON. It is used for summaries and memory passes, so reliability beats creativity.
- Embedding model: consistent vector space and stable API support. Latency and throughput matter because retrieval happens often.

If a model struggles with JSON-only responses, add it to `json_only_disabled_models` to relax strict JSON enforcement for that model.

---

## Evidence and Provenance

Evidence IDs are attached to internal signals, memory writes, and tool outputs. Self-claims and memory writes without evidence are blocked or marked provisional. User-visible answers are instructed to cite evidence or mark uncertainty when evidence is thin.

The result is slower assertion but stronger auditability. You can trace why a claim exists and where it came from.

---

## Memory as a Knowledge Language

Memory is a DSL with explicit grammar for facts, relations, time, confidence, and source references. This makes memory both queryable and correctable.

When beliefs conflict, the system records a conflict set rather than overwriting. This preserves truth while the operator resolves uncertainty.

See:
- `docs/ics_v4_1_dsl.md`
- `memory_syntax.md`

---

## Prompt Discipline and Context Control

The prompt builder measures section sizes, trims when needed, and protects anchor sections. Context hydration modes control how much memory and context enter the prompt. The system records trimming events so you can see when and why context was dropped.

This makes behavior predictable under load instead of silently brittle.

---

## Self-Modeling and Internal Signals

Symbiote tracks internal signals such as qualia tags, wave coherence, and self-model confidence. These are telemetry, not theater. They can influence arbitration and are logged so an operator can inspect how the system arrived at a decision.

---

## Tools Are Governed Capabilities

Tools are registered with explicit capability levels and risk profiles. The kernel validates tool names and arguments before execution and logs decisions. Higher risk tools require stronger gating. This keeps the system honest about what it can do.

---

## Observability and Auditability

System logs are structured and stored in SQLite. You can inspect runs, tool calls, memory passes, and health snapshots.

Tables of interest include:
- `system_logs`
- `memory_write_ledger`
- `episodic_events`
- `ics_fact_beliefs`
- `ics_rel_beliefs`

Default DB location (Windows, Tauri):
`C:\Users\<you>\AppData\Roaming\com.symbiote.app\symbiote.db`

---

## Background Continuity

The scheduler keeps summaries, memory consolidation, and health snapshots updated in the background. This prevents long-term drift and keeps the system coherent across sessions.

---

## What It Is Not

- It does **not** claim consciousness or subjective experience.
- It is **not** optimized for minimal latency.
- It is **not** an unconstrained agent; tools are gated by design.
- It is **not** a black box; the system is meant to be observable.

---

## Quick Start

Requirements: Node.js, Rust toolchain, Tauri CLI, Python 3.

```bash
python scripts/bootstrap.py
```

```bash
npm run tauri dev
python voice_service_v2.py
```

Optional one-shot run:

```bash
python scripts/bootstrap.py --run
```

---

## Core Paths

```
src-tauri/src/core/
|-- kernel/              (orchestration, arbitration, gating, commit)
|-- memory/              (DSL, candidates, compiler, writer, retrieval)
|-- self_memory/         (self-memory bridge + telemetry)
|-- prompt_builder.rs
|-- model_client.rs
|-- scheduler.rs
|-- system_log.rs

src/
|-- views/               (ChatView, TraceView, SettingsView)
|-- components/          (MemoryGraph3D, SystemStatePanel, VoiceController)
|-- utils/
```

---

## Status

Active and experimental. The architecture is coherent, but reliability depends on model quality, evidence signal quality, and operational discipline. Expect iteration.

---

## Screenshots

Chat + system state:

![Symbiote chat view](docs/screenshots/chat-view.png)

Memory graph:

![Symbiote memory graph](docs/screenshots/memory-graph.png)
