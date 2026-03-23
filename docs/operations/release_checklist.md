# Release Checklist

## Required Artifacts
1. Latest baseline report: `reports/latest_baseline.json`.
2. Kernel pipeline regression: `reports/kernel_pipeline_regression.json`.
3. Cognition scorecard: `reports/latest_scorecard.json`.

## Release Gates
1. Outcome accuracy >= 0.70 over the last scorecard window.
2. Memory drift events <= 2 over the last scorecard window.
3. Telemetry drift events <= 1 over the last scorecard window.
4. Tool failure rate <= 0.25 over the last baseline window.
5. No unresolved `Audit` gate notices in the last 20 decisions.

## Commands
1. `python scripts/baseline_runner.py --minutes 180`
2. `python scripts/kernel_pipeline_regression.py --legacy-minutes 120 --phased-minutes 120`
3. `python scripts/cognition_scorecard.py --minutes 180`

## If A Gate Fails
1. Switch `memory_write` and `memory_consolidation` to `degraded` for drift issues.
2. Switch `tool_execution` to `degraded` for tool reliability issues.
3. Document the mitigation in the release notes.
