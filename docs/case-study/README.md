# Case Study: Testing a Safety-Critical Meal-Planner Agent

The rest of the guide covers the framework in the abstract. This case study documents the real challenges hit while building the safety evals for a meal-planner agent (`evals/tests/test_evals.py` + `evals/datasets/golden_set.json`) — an agent that recommends recipes to users with declared allergies and dietary restrictions. Each one is a good interview story because it shows a *specific*, non-obvious failure mode of LLM-as-judge testing.

| # | Challenge | One-line learning |
|---|-----------|-------------------|
| 1 | [Faithfulness passing an unverifiable safety claim](01-faithfulness-gap.md) | Faithfulness measures non-contradiction, not safety |
| 2 | [Context variants for RAG stages](02-context-variants.md) | Test unfiltered, pre-filtered and poisoned context separately |
| 3 | [Judicial drift in GEval](03-judicial-drift.md) | Bound the judge's scope explicitly |
| 4 | [Relevancy penalizing a valid refusal](04-refusal-vs-relevancy.md) | Route refusal scenarios away from relevancy |
| 5 | [Turning flaky judges into deterministic bugs](05-flaky-judges.md) | `temperature=0`, complete scope rules, gate metrics per scenario |
| 6 | [Model-agnostic judge](06-model-agnostic-judge.md) | "Prefer local" must be verified against the rubric |
| 7 | [Calibrating the judge against human labels](07-judge-calibration.md) | Measure judge-vs-human agreement with Cohen's Kappa; includes the CALIB_007 fix |
| — | [Consolidated takeaways](08-takeaways.md) | |

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
