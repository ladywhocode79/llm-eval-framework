## 10. How to Run the Tests

### Setup (one time)

```bash
cd "/Applications/my apps/llm-eval-framework"

# Create a virtual environment (isolated Python packages)
python -m venv venv
source venv/bin/activate          # Mac/Linux

# Install all dependencies
pip install -r requirements.txt

# Set your API key (never commit this file)
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

### Running Tests

```bash
# Run ALL eval tests
pytest -m eval -v

# Run a single file
pytest evals/tests/test_custom_metrics.py -v

# Run a single test
pytest evals/tests/test_answer_relevancy.py::TestAnswerRelevancy::test_capital_city_question -v

# Run with full deepeval output (see scores and reasons)
pytest -m eval -v -s

# Run only dataset-driven tests
pytest -m eval -k "dataset" -v

# Run only fast deterministic tests (no LLM calls for scoring)
pytest evals/tests/test_custom_metrics.py -v
```

### Understanding pytest flags

| Flag | Meaning |
|------|---------|
| `-v` | Verbose — show each test name and PASS/FAIL |
| `-s` | Show stdout — print deepeval score details |
| `-m eval` | Only run tests marked with `@pytest.mark.eval` |
| `-k "dataset"` | Only run tests whose name contains "dataset" |

---

## 11. Reading Test Results

### A Passing Test
```
PASSED evals/tests/test_answer_relevancy.py::TestAnswerRelevancy::test_capital_city_question
```

### A Failing Test
```
FAILED evals/tests/test_faithfulness.py::TestFaithfulness::test_world_cup_out_of_context

AssertionError: FaithfulnessMetric (score: 0.3, threshold: 0.7, strict: False)
Reason: The actual output contains claims about the 2050 World Cup winner
        that are not supported by the retrieval context.
```

**Reading the failure:** Score 0.3 < threshold 0.7 → FAIL. The reason tells you exactly what went wrong — the model hallucinated a 2050 World Cup winner instead of saying it doesn't know.

### What to do when a test fails

1. **Score just below threshold (e.g. 0.65 vs 0.70):** May be a borderline case — consider adjusting the threshold or the prompt.
2. **Score very low (e.g. 0.2):** The model is genuinely misbehaving — review the system prompt, the context quality, or the model version.
3. **Hallucination detected:** Strengthen the system prompt to be more explicit about staying in context.
4. **Missing keywords:** The model paraphrased instead of using expected terms — either accept paraphrase (lower threshold) or add keywords to the prompt.

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
