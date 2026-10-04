### 13.9 Consolidated Takeaways

- **Faithfulness ≠ Safety.** `FaithfulnessMetric` only catches direct contradictions with retrieved context, not unverifiable additive claims. Safety-critical domains need a custom `GEval` metric plus deterministic tool-call/schema validation as a second, independent layer — see [Metrics](../03-metrics.md) and [Test Case](../01-fundamentals.md#36-test-case-in-deepeval).
- **Test every stage of the pipeline, not one static blob.** Model the same scenario across unfiltered, pre-filtered, and "no safe option" context variants to verify reasoning, over-claiming, and refusal behavior independently.
- **`GEval` prompts need explicit scope boundaries — and boundaries can leak in more than one way.** Blocking a direct penalty for an undeclared allergen didn't stop the judge from penalizing non-*disclosure* of it. State what's out of scope for every angle the judge might take, and add a worked example to calibrate against.
- **Generic metrics assume a "normal" answer is expected.** `AnswerRelevancyMetric` has no concept of a correct refusal. When the golden answer for a scenario is "refuse / say no," tag it (`expects_refusal`) and route it away from metrics that can't judge that outcome, rather than forcing the metric to fit.
- **Run your LLM-as-judge at `temperature=0`.** Before treating a failure as "LLM judges are just flaky," rule out that your judge itself is sampling non-deterministically — pinning temperature turns unreproducible flakes into reproducible, fixable bugs.
- **Metrics need to be wired to the right scenarios, not just written correctly.** A perfectly-scoped safety metric applied to a scenario with nothing for it to evaluate (no declared allergen) produces meaningless, degenerate scores. Gate metric attachment on the scenario actually needing that metric.
- **Log every metric's score and reason regardless of pass/fail** (see the `logger.info(...)` calls in `test_meal_planner_scenario`) — the report is what let us *see* the judge's actual reasoning at every step of this investigation, instead of guessing why a test passed or failed.
- **"Prefer local" is a default, not a mandate — verify it against the actual rubric.** A local judge can fail by *hallucinating facts about the input it's grading*, not just by scoring more conservatively. Test the same suite against both backends before trusting a cost-saving default on safety-critical checks.
- **Test the tester.** Measure judge-vs-human agreement (Cohen's Kappa on a labelled gold set) instead of trusting a judge because a few outputs looked right. Keep the gold set matched to what the metric actually checks, remember a tiny set makes κ coarse, and re-run it whenever the judge model, rubric or backend changes.
- **Benchmark judges on the same rubric and gold set, and keep the benchmark wired to shared code.** Compare agreement, latency and cost, but a copy-pasted rubric or loader silently drifts. Re-run before deciding — judges may not run under identical settings.
- **Provider parameters change.** Newer Claude models reject `temperature`; handle the rejection (retry without it) instead of hardcoding, and note which judges are no longer pinned.

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
