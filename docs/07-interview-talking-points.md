## 12. Interview Talking Points

These are key things to highlight when discussing this project in an SDET interview.

### "What makes LLM testing different from regular API testing?"

> "Traditional API testing uses exact assertions — status codes, field values, schema validation. LLM testing is fundamentally different because the outputs are non-deterministic. The same question can produce slightly different answers each time. So instead of exact matching, we use quality metrics — scores between 0 and 1 — and define a threshold for what counts as acceptable. This requires a completely different mindset: instead of 'is this exactly right,' we ask 'is this good enough?'"

---

### "What is LLM-as-judge and why is it useful?"

> "LLM-as-judge is using a separate AI model to evaluate the quality of another AI's output. For example, we use Claude to score whether our QA pipeline's responses are relevant or faithful. The reason it's powerful is that semantic quality is hard to measure with simple rules — 'Paris is the capital' and 'The capital is Paris' are equivalent but wouldn't match exactly. An LLM judge understands meaning. The trade-off is cost and speed — every evaluation triggers an additional API call."

---

### "How do you handle hallucination testing?"

> "We test for hallucination using the FaithfulnessMetric from deepeval. It compares the model's output against the provided context documents and scores what percentage of claims in the output are actually grounded in the context. We also built a custom deterministic metric — NoHallucinatedNumberMetric — that uses regex to extract all numbers from the output and checks that every number also appears in the context. Numbers are particularly dangerous to hallucinate because they sound authoritative."

---

### "What is the difference between deterministic and LLM-based metrics?"

> "Deterministic metrics are pure Python logic — keyword checks, length validations, regex patterns. They're fast, free, and 100% consistent. LLM-based metrics use another AI call to judge quality semantically. They're slower and cost API tokens but can evaluate nuanced properties like relevance and faithfulness. In a CI pipeline strategy, I'd run deterministic checks on every commit and LLM-based evals on a schedule or before releases."

---

### "How is the test data managed?"

> "We separate test data from test logic using a JSON dataset file. Test cases have IDs, categories, tags, and expected outputs. Pytest parametrize reads from this dataset to run multiple cases through the same test function. This means a non-technical team member can add test cases by editing a JSON file without touching any Python code. It also makes the tests data-driven, which is a core best practice in test automation."

---

### "How do you manage the cost of LLM-based evaluations?"

> "By default our framework uses a local model via Ollama as the LLM judge instead of OpenAI. We subclassed deepeval's DeepEvalBaseLLM to wrap Ollama's Python library, then injected it into metrics via a pytest fixture controlled by an environment variable. This means every local test run costs nothing for evaluation. We keep OpenAI as an option for pre-release quality gates where higher accuracy is worth the cost — you just flip the JUDGE_BACKEND env var. The app under test still uses Claude's API, but the evaluator is free."

---

### "How does the framework scale?"

> "The current framework can scale in a few ways: adding more test cases to the JSON dataset without code changes, adding new metrics by subclassing BaseMetric, integrating with CI/CD to run on every deployment, connecting to a real vector database for genuine RAG testing, and adding more metrics like toxicity or bias detection. The architecture keeps the app and evals separate, so we can point the same eval framework at different LLM backends."

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
