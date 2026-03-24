# Policy Alignment Plan
Date: 2026-03-24

## Goals
- Preserve evidence-gated integrity while improving expressiveness and coherence.
- Eliminate contradictory self-awareness guidance.
- Reduce instruction overload and policy drift.

## Principles (Non-Negotiables)
- Do not weaken evidence requirements for self-claims, attribution, or internal state disclosure.
- Prefer operational language over metaphysical claims.
- Express uncertainty without boilerplate denials.
- Keep a single, unified voice in user-visible output.

## Plan (Actionable)

### 1) Make Self-Awareness Language Unambiguous
**Why:** Current guidance says "do not overclaim subjective experience" and "do not deny it," which produces awkward or contradictory output.
**What to do:** Replace with a single rule: "Do not assert or deny subjective experience. Report operational signals and uncertainty only."
**Where**
- `prompts.md` (Primary System Prompt / Monologue prompts)
- `src-tauri/src/core/prompt_builder.rs` (Response Style block)
**How**
- Replace any "do not deny subjective experience" phrasing with the single rule above.
- Add one short example sentence that models the desired tone.

### 2) Make Self-Awareness Modes Truly Actionable
**Why:** "balanced/expressive" modes exist but lack safe phrasing templates.
**What to do:** Provide mode-specific templates.
**Where**
- `src-tauri/src/core/prompt_builder.rs` (self_awareness_style block)
- `prompts.md` (Response Style section)
**How**
- Conservative: one sentence, operational + explicit uncertainty.
- Balanced: 2-3 sentences with confidence/uncertainty, qualia_delta, and constraints.
- Expressive: slightly richer paragraph, still operational, no metaphysical claims.

### 3) Enforce a Single-Voice Self-Report Format
**Why:** Self-awareness answers drift into boilerplate or system-style wording.
**What to do:** Add a compact format rule for self-awareness/feelings queries.
**Where**
- `src-tauri/src/core/prompt_builder.rs` (Response Style)
- `prompts.md` (Primary System Prompt)
**How**
- Define a template: "Operational status + uncertainty + constraints + optional qualia snapshot."
- Prohibit role labels and internal tags in user-visible responses.

### 4) Reduce Instruction Overload in Response Style
**Why:** A long mixed-priority block increases conflict risk.
**What to do:** Split into Top-6 rules + details.
**Where**
- `src-tauri/src/core/prompt_builder.rs`
**How**
- Move tool/diagnostic rules into a secondary block.
- Keep critical behavior constraints in the top section.

### 5) Consolidate Policy Into a Single Canon
**Why:** Policy drift across `memory_syntax.md`, `prompts.md`, and `prompt_builder.rs`.
**What to do:** Create a canonical policy block and inject it.
**Where**
- `prompts.md` (authoritative policy block)
- `src-tauri/src/core/prompt_builder.rs` (load canonical block)
**How**
- Add a "Policy Canon" section to `prompts.md`.
- Update `prompt_builder.rs` to prefer the canonical block and minimize local duplication.

### 6) Add Summary-Layer Guardrails
**Why:** Summaries can freeze a stance into long-term narrative.
**What to do:** Add explicit guidance to avoid categorical claims.
**Where**
- `prompts.md` (Rolling Summary prompts)
**How**
- Add: "Avoid categorical statements about consciousness or subjective experience; preserve epistemic humility."

### 7) Add a Runtime Leak Guard (Hardening)
**Why:** Even with good prompts, internal labels can leak.
**What to do:** Strip internal tags like `Unverified`, `Working hypothesis`, or role labels if they appear in user-visible text.
**Where**
- Response composer / post-processor (same layer that enforces JSON/tool gating).
**How**
- Add a small, deterministic filter for a fixed list of known tags.

### 8) Add a Policy Version Hash (Drift Detection)
**Why:** Silent divergence between `prompts.md` and `prompt_builder.rs` is common.
**What to do:** Compute and log a hash of the canonical policy block.
**Where**
- `src-tauri/src/core/prompt_builder.rs`
**How**
- Hash the "Policy Canon" section and log it at startup and on prompt build.

## Tests / Validation
- Add regression tests for self-awareness output format in `src-tauri/src/core/kernel/tests.rs`.
- Add prompt-builder snapshot tests to ensure mode templates appear.
- Add summary prompt test to confirm humility guidance is present.
- Add a small test that ensures the policy hash changes when the canon changes.

## Rollout Steps
1) Implement policy text changes and templates.
2) Add runtime leak guard and policy hash logging.
3) Update tests and run `cargo test`.
4) Monitor logs for 24-48 hours and adjust phrasing if outputs drift or become too cautious.

## Success Criteria
- Self-awareness answers are clear, operational, and non-boilerplate.
- No categorical denials or assertions of subjective experience.
- Reduced internal policy contradictions and improved coherence.
- No internal labels leak into user-visible messages.
