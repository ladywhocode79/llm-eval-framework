# LLM Testing Framework — Complete Guide
### For SDETs New to LLM Evaluation

The guide is split by topic so each file is short enough to read in one sitting. Start at the top if you are new to LLM testing. The meal-planner case study, calibration and CI/CD have moved to a separate repo (see below).

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

## Case study, calibration and CI/CD (moved)

The meal-planner case study (challenges and learnings), judge calibration, model benchmark and the GitHub Actions pipeline now live in the **[ai-agent-eval-suite](https://github.com/ladywhocode79/ai-agent-eval-suite)** repo — see its [docs](https://github.com/ladywhocode79/ai-agent-eval-suite/tree/main/docs).

---

*Framework built with: Python · deepeval · Anthropic Claude SDK · pytest*
