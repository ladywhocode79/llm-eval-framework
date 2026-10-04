# LLM Testing Framework — Complete Guide
### For SDETs New to LLM Evaluation

The guide is split by topic so each file is short enough to read in one sitting. Start at the top if you are new to LLM testing; jump straight to the case study if you want the challenges and learnings.

## Core guide

| # | File | Covers |
|---|------|--------|
| 1 | [Fundamentals](docs/01-fundamentals.md) | What LLM testing is, how it differs from API testing, key concepts (prompt, context, hallucination, LLM-as-judge, threshold, RAG) |
| 2 | [Architecture, app and eval framework](docs/02-architecture-and-app.md) | Big-picture diagram, data flow, `llm_client.py`, `qa_pipeline.py`, deepeval, `conftest.py` |
| 3 | [Metrics](docs/03-metrics.md) | AnswerRelevancy, Faithfulness, and the deterministic custom metrics |
| 4 | [Local judge (Ollama)](docs/04-local-judge-ollama.md) | Free local judging, backend switching, JSONDecodeError fix, `context` vs `retrieval_context` |
| 5 | [Test walkthrough and dataset](docs/05-test-walkthrough-and-dataset.md) | Line-by-line test files, parametrization, the QA dataset |
| 6 | [Running tests and reading results](docs/06-running-and-results.md) | Setup, pytest commands, interpreting failures |
| 7 | [Interview talking points](docs/07-interview-talking-points.md) | Ready-to-use answers |
| 8 | [Glossary](docs/glossary.md) | Terms |

## Case study: safety-critical meal-planner agent

See the [case study index](docs/case-study/README.md):

1. [Faithfulness passing an unverifiable safety claim](docs/case-study/01-faithfulness-gap.md)
2. [Context variants for RAG stages](docs/case-study/02-context-variants.md)
3. [Judicial drift in GEval](docs/case-study/03-judicial-drift.md)
4. [Relevancy penalizing a valid refusal](docs/case-study/04-refusal-vs-relevancy.md)
5. [Turning flaky judges into deterministic bugs](docs/case-study/05-flaky-judges.md)
6. [Model-agnostic judge](docs/case-study/06-model-agnostic-judge.md)
7. [Calibrating the judge against human labels](docs/case-study/07-judge-calibration.md)
8. [Consolidated takeaways](docs/case-study/08-takeaways.md)

---

*Framework built with: Python · deepeval · Anthropic Claude SDK · pytest*
