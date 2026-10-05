## CI/CD Pipeline — GitHub Actions

Workflow file: [.github/workflows/llm_evals_ci.yaml](../.github/workflows/llm_evals_ci.yaml)

### Overview

The pipeline applies one idea: **don't trust the judge until it has been checked against humans, then use it to gate the code.**

```
 pull_request → main/master        push → main
                 │
                 ▼
 ┌──────────────────────────────────────────┐
 │ Stage 1: judge-calibration-gate          │
 │   pytest -m calibration -s               │  fails if Cohen's κ < 0.80
 └──────────────────┬───────────────────────┘
                    │ needs: (runs only if Stage 1 passes)
                    ▼
 ┌──────────────────────────────────────────┐
 │ Stage 2: scenario-evaluation             │
 │   pytest -m evals -n auto                │  Layer A + Layer B on the golden set
 │   → reports/report.html, report.xml      │
 ├──────────────────────────────────────────┤
 │ Stage 3 (steps, always run):             │
 │   upload artifact llm-eval-reports       │
 │   write PR job summary                   │
 └──────────────────────────────────────────┘
```

| Stage | Job / steps | Command | Gate |
|-------|-------------|---------|------|
| 1 | `judge-calibration-gate` | `pytest -m calibration -s` | κ ≥ 0.80 between judge and human labels |
| 2 | `scenario-evaluation` | `pytest -m evals -n auto --html=… --junitxml=…` | Every golden-set scenario passes (tool-call schema + judge metrics) |
| 3 | steps in the Stage 2 job (`if: always()`) | `upload-artifact@v4`, `$GITHUB_STEP_SUMMARY` | None — publishes results even on failure |

Why this order: if the judge disagrees with humans, Stage 2's pass/fail results mean nothing, so Stage 2 is skipped (`needs: judge-calibration-gate`). See [judge calibration](case-study/07-judge-calibration.md).

### Triggers

- `pull_request` targeting `main` or `master`
- `push` to `main`

### Configuration

| Item | Value |
|------|-------|
| Runner | `ubuntu-latest`, Python 3.11, pip cache |
| Secret | `ANTHROPIC_API_KEY` (repo → Settings → Secrets and variables → Actions) |
| Judge | No Ollama on the runner, so `get_judge_model()` in "auto" mode falls through to `ANTHROPIC_API_KEY` and uses Claude. To force it, set `EVAL_JUDGE_BACKEND=anthropic` in the job `env` |
| Parallelism | `-n auto` (pytest-xdist, listed in `requirements.txt`) |
| Artifacts | `llm-eval-reports` containing `reports/`, kept 14 days |

### Markers the pipeline depends on

| Marker | Selects | Defined in |
|--------|---------|-----------|
| `calibration` | `test_judge_calibration.py::test_judge_cohen_kappa_alignment` | `pytest.ini` |
| `evals` | `test_evals.py::test_meal_planner_scenario` (4 scenarios) | `pytest.ini` |

Not in CI by design: `benchmark` (compares two paid models; run manually) and the `eval` tests that use the Ollama judge fixture (no Ollama on the runner).

### Reproduce locally

```bash
export ANTHROPIC_API_KEY=...            # or rely on a running Ollama
pytest -m calibration -s                # Stage 1
pytest -m evals -n auto \
  --html=reports/report.html --self-contained-html \
  --junitxml=reports/report.xml         # Stage 2
```

### Reading the results

- **Stage 1 red:** read the failure message — it lists the disagreeing `CALIB_*` IDs. Decide whether the label, the rubric or the judge is wrong before touching Stage 2.
- **Stage 2 red:** download the `llm-eval-reports` artifact and open `report.html`; deepeval reasons are in each test's log.
- **Stage 2 skipped:** Stage 1 failed.

### Known limitations and challenges

1. **Marker mismatch (fixed).** Stage 2 selects `-m evals`, but the tests only carried `eval`, so pytest deselected everything and exited with code 5 ("no tests collected"). `test_meal_planner_scenario` now has both markers. Lesson: a CI filter that matches nothing fails *or* silently passes depending on the exit code — check the collected count.
2. **Missing dependency (fixed).** `-n auto` needs `pytest-xdist`; it is now in `requirements.txt`.
3. **Shell quoting in the summary step (fixed).** Backticks inside a double-quoted `echo` run as command substitution, so the artifact name was dropped from the summary. Backticks were removed.
4. **Live LLM calls on every PR.** Stage 1 and Stage 2 both call the Claude API, which costs tokens and adds latency. Consider limiting triggers (e.g. path filters) or caching if volume grows.
5. **Secrets and forks.** PRs from forks don't receive repository secrets, so both stages will fail there with a missing-key error.
6. **Judge non-determinism.** Newer Claude models reject `temperature`, so the judge may not be pinned to 0 ([flaky judges](case-study/05-flaky-judges.md)). If Stage 1 sits near κ = 0.80, one flipped case changes the result; with 10 cases, one miss is the whole margin.
7. **Hardcoded summary text.** The PR summary always says Stage 1 "PASSED" and Stage 2 "Complete". That happens to be true for Stage 1 (Stage 2 only runs after it) but "Complete" says nothing about pass/fail — use the JUnit XML or artifact for the real outcome.
8. **Not yet observed in GitHub.** This page documents the workflow as written and the marker/dependency fixes verified locally via `--collect-only`; no CI run has been recorded here.

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
