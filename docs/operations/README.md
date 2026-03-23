# Operations Playbook

## Purpose
This playbook explains how to operate Symbiote using the new trust indicators, outcome adjudication tools, and health signals. It is designed for day-to-day operators and release managers.

## Trust Signals (Chat UI)
1. `Grounded` means evidence IDs are present and the response is not speculative. You can act on it with normal caution.
2. `Partial` means evidence exists but the response is marked speculative. Confirm before acting.
3. `Speculative` or `Unverified` means no evidence IDs were attached. Request clarification or trigger a tool run before acting.
4. `Audit` means the kernel flagged the response for review. Prefer a quick evidence check before proceeding.

## Outcome Adjudication (Cockpit)
1. Open the System Cockpit > Diagnostics.
2. Use the Outcome Adjudication panel to confirm, disconfirm, or mark outputs inconclusive.
3. Always attach outcomes to the closest relevant assistant message.
4. Confirmed outcomes reinforce memory confidence. Disconfirmed outcomes automatically decay memory confidence.

## Drift Response
1. If memory drift events appear in System Health, pause new memory writes and review recent memory edits.
2. Run `scripts/cognition_scorecard.py` to validate outcome accuracy and drift trends.
3. If drift persists, set `memory_write` and `memory_consolidation` to `degraded` until drift clears.

## Degraded Modes
1. `memory_write` degraded: limits new memory writes and slows consolidation.
2. `tool_execution` degraded: reduces tool calls; the kernel will refuse low-confidence tool usage.
3. `self_memory` degraded: prevents unreliable self-claims from persisting.

## Recovery Checklist
1. Verify outcome accuracy is stable for at least one scorecard window.
2. Ensure memory drift events are zero in the last window.
3. Return degraded subsystems to `normal` only after evidence coverage recovers.
