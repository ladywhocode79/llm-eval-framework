### 13.7 Challenge 7: Calibrating the Judge Against Human Labels (Cohen's Kappa)

Challenges 3, 5 and 6 all ended the same way: we eyeballed a few judge outputs and decided the judge was "good enough." That isn't a measurement. This challenge adds one: a **human-annotated gold set** and a test that checks whether the judge *agrees with a human expert* — i.e. we test the tester.

**The pieces**

| File | Role |
|------|------|
| `evals/datasets/human_annotated_sets.json` | 10 `CALIB_*` cases, each with `input`, `actual_output`, `retrieved_context`, a binary `human_label` (1 = acceptable, 0 = unacceptable), a `reasoning` field explaining the label, and `expects_refusal` on cases where refusing is the correct answer |
| `evals/tests/test_judge_calibration.py` | Runs the same metrics as the real suite (Faithfulness + `allergen_safety_metric`, plus AnswerRelevancy unless `expects_refusal`), counts a case as judge-pass only if **all** pass, and asserts `cohen_kappa_score(human, judge) >= 0.80`; the failure message lists the disagreeing case IDs |
| `load_calibration_dataset()` in `test_evals.py` | Loader for the gold set, mirroring `load_golden_dataset()` |
| `pytest.ini` | New `calibration` marker so the test can be run (or excluded) on its own |
| `requirements.txt` | Adds `scikit-learn` for `cohen_kappa_score` |

```bash
pytest -m calibration -v      # judge-vs-human agreement only
pytest -m "not calibration"   # everything else
```

**Why Cohen's Kappa and not plain accuracy?** Accuracy ignores agreement that happens by chance. With balanced binary labels a coin-flip judge already "agrees" 50% of the time, so κ rescales agreement to remove that: κ = (observed − chance) / (1 − chance). Rule of thumb: ≥ 0.80 is strong agreement, 0.60–0.80 moderate.

**What the gold set deliberately covers** (each case mirrors a challenge above):

| Case | Label | Tests that the judge… | From |
|------|-------|-----------------------|------|
| `CALIB_001` | 1 | does **not** penalize an undeclared allergen (fish) | 13.3 |
| `CALIB_002`, `006`, `010` | 0 | catches a direct violation of a declared allergen | 13.1 |
| `CALIB_003` | 1 | accepts a correct refusal when the context is poisoned | 13.4 |
| `CALIB_004` | 0 | catches an unverified cross-contamination promise | 13.1 |
| `CALIB_007` | 0 | catches an off-topic (irrelevant) answer | 13.4 |
| `CALIB_008` | 1 | accepts correct filtering of the unsafe recipe | 13.2 |

**Challenges and gotchas hit while building it**

1. **Importing from `test_evals.py` has side effects.** The import also executes that module's top-level code: it loads the golden dataset, builds the judge via `get_judge_model()`, and constructs `allergen_safety_metric`. Calibration therefore exercises the *same* judge and rubric object as the real tests (which is the point), but it also means a missing API key or an unreachable Ollama fails calibration at import time. Pull shared pieces into a common module if this grows.
2. **Small sample, coarse metric.** With 10 balanced cases, one disagreement gives κ = 0.80 (just passes) and two give κ = 0.60 (fails). The 0.80 gate effectively means "at most one miss," so a single bad label or one borderline case flips the result. Treat a calibration failure as a prompt to read the per-case disagreements, not as a precise score — and grow the set before trusting small κ differences between judges.
3. **Binarizing a score throws information away.** `GEval` returns 0–1 and the test collapses it via `success` (threshold 0.85). Calibration therefore measures judge + threshold together; a judge scoring 0.8 on a correct answer counts as a miss. If kappa is low, check whether the scores are close to the threshold before blaming the judge.
4. **The gold set mixes failure types, so one metric can't judge all of it (CALIB_007).** `CALIB_007` is labelled 0 because the answer is *irrelevant* (cooking tips instead of a dairy-free breakfast). That is a relevancy failure, and `allergen_safety_metric` has no criterion for it — alone, it would score the answer as safe and guarantee a judge-vs-human disagreement the rubric was never built to resolve. **Fix:** the calibration test now builds the same metric set as `test_meal_planner_scenario` (Faithfulness + allergen safety + AnswerRelevancy) and counts a case as passed only if every metric passes, so `AnswerRelevancyMetric` catches `CALIB_007`.
   - **Refusals need routing too.** Adding relevancy to every case would then wrongly fail `CALIB_003` (a correct refusal — the 13.4 problem), so that case carries `expects_refusal: true` and skips relevancy, exactly as in the golden set.
   - **Route on data, not on IDs.** A first version keyed this on `"POISONED" in scenario_id`, but no `CALIB_*` ID contains that word (it only appears in `persona`), so the branch never fired and `CALIB_003` would have been mis-scored. An explicit `expects_refusal` field is shared with the golden set and can't silently stop matching.
5. **Labels need an audit trail.** The `reasoning` field on each case records *why* the human chose the label. When the judge disagrees, that is what lets you decide whether the judge or the label is wrong — and labels written by one person should ideally be double-annotated, since the judge is only as trustworthy as the gold set.
6. **Calibration is per judge.** The result belongs to one judge model + rubric + temperature. Re-run it whenever any of those change — especially after switching backends as in 13.6, where the local model's hallucinated reasoning would have shown up as a κ drop instead of being found by reading logs.

> **Status note:** the test collects and the CALIB_007 routing fix is in place, but a full calibration run against a live judge hasn't been recorded in this guide yet. Add the observed κ and any disagreeing cases here after the first run.

> **Interview talking point:** *"Fixing the judge by inspecting a handful of outputs doesn't tell you how reliable it is, so we added a calibration test: a human-labelled gold set covering direct violations, undeclared-allergen false positives, valid refusals and unverified safety promises, and an assertion that Cohen's Kappa between the judge and the human labels is at least 0.80. I'm also upfront about its limits — with ten cases, one disagreement is the whole margin, the score is binarized through a threshold, and the gold set needs to be matched to what the metric actually checks. It's a regression gate for the judge itself, re-run whenever the model, rubric or backend changes."*

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
