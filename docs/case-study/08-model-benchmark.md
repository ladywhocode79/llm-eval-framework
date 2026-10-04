### 13.8 Challenge 8: Benchmarking Judge Models on Accuracy, Latency and Cost

Calibration (13.7) tells you whether *a* judge agrees with humans. The next question is *which* judge to use: a cheaper, faster model, or a stronger, pricier one. `evals/tests/test_model_benchmark.py` answers it by running the same calibration set through two Claude judges and comparing them side by side.

**What it measures** (per model, over the 10-case human-labelled set):

| Metric | How |
|--------|-----|
| Accuracy | Cohen's Kappa between judge verdicts and human labels (same ≥ 0.80 gate as 13.7) |
| Latency | p50 / p95 per evaluated case, plus total wall-clock time |
| Cost | Token count × per-token price. Token counts are **fixed estimates** (350 in / 150 out per case), not measured usage |

```bash
pytest -m benchmark -v -s     # live calls to both models; also writes reports/model_benchmark.md and .json
```

The two models are `claude-haiku-4-5-20251001` and `claude-sonnet-5-5`. The same `allergen_safety_metric` rubric is reused for both, so the only variable is the judge.

**Challenges and gotchas**

1. **The benchmark drifted from the code it benchmarks.** It first shipped importing `ClaudeLLM` from `test_evals.py` (it had moved to `app/judge_factory.py`), used a missing `json` import, pointed at a misspelled dataset filename, targeted retired Claude 3 model IDs, and carried its own shortened copy of the rubric. A benchmark with a different rubric from the real suite measures nothing useful. **Fix:** import the shared judge class, dataset loader and `allergen_safety_metric.criteria` so the benchmark can't diverge.
2. **Newer models reject `temperature`.** `claude-sonnet-5-5` returns `400 invalid_request_error: temperature is deprecated for this model`, which broke the benchmark mid-run. `ClaudeLLM` now sends `temperature=0` first and, if the API rejects it, drops the parameter, retries once, and stops sending it for that instance (sync and async paths).
3. **Fixing it creates a fairness caveat.** Haiku is pinned to `temperature=0`; Sonnet 5.5 now runs at the model default. Sonnet's verdicts can vary slightly between runs (the 13.5 problem), so a single run's κ is weaker evidence for Sonnet than for Haiku. Run the benchmark more than once before choosing a judge.
4. **The benchmark's metric is narrower than the calibration test's.** It scores only `allergen_safety_metric`, while 13.7 also routes Faithfulness and AnswerRelevancy. `CALIB_007` (off-topic answer, labelled 0) is a relevancy failure, so both models will likely "miss" it — which spends the single disagreement that κ ≥ 0.80 allows on 10 cases. Compare models on *which* cases they disagree on, not just the κ number.
5. **Cost is an estimate with an assumed price.** The Haiku 4.5 rate ($1 / $5 per million tokens) is known; the Sonnet 5.5 rate in the file is an assumption and must be checked against current pricing. Use the cost row to compare orders of magnitude, not to budget.
6. **Tiny sample, noisy latency.** Ten cases make p95 effectively the slowest call, and latency includes network variance and deepeval overhead.

**First observed run** (from `reports/report.xml`, 2026-10-04):

| Result | Value |
|--------|-------|
| Test outcome | **Failed** the accuracy gate |
| Haiku 4.5 κ | **0.60** (< 0.80) — on 10 balanced cases, that is two disagreements with the human labels |
| Wall-clock for both models | ~106 s |
| Sonnet 5.5 κ, latency, cost | **Not captured** |

Why Sonnet's numbers are missing: the comparison table is printed to stdout, which `report.html` / `report.xml` don't record, and the test asserted Haiku's κ before checking Sonnet. Two fixes were made: the test now writes `reports/model_benchmark.md` and `reports/model_benchmark.json` after both models finish, and it asserts on both models together, so one model's failure can't hide the other's numbers.

What κ = 0.60 means: this is the same size of miss as the calibration test (13.7), so it is a signal to inspect the disagreeing cases, not a verdict on Haiku. One likely cause is `CALIB_007` (an off-topic answer the safety-only metric can't judge); the second miss has not been identified. Next step: re-run, then compare *which* cases each model misses.

> **Interview talking point:** *"Once the judge was calibrated against human labels, I benchmarked a cheaper and a stronger model on the same gold set and rubric, comparing agreement, latency and cost rather than assuming bigger is better. Along the way I hit a real compatibility issue — the newer model rejects the temperature parameter — which meant the two judges weren't run under identical settings, so I treat a single run as indicative and re-run before deciding."*
