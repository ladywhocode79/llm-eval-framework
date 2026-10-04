## 8. Test Files — Line by Line Walkthrough

### `test_answer_relevancy.py` — Full Walkthrough

```python
@pytest.mark.eval                          # Custom marker — run with: pytest -m eval
class TestAnswerRelevancy:

    THRESHOLD = 0.7                        # 70% relevancy required to pass

    def _run(self, pipeline, question, context=""):
        result = pipeline.answer(          # Step 1: Call the LLM app
            question=question,
            context=context
        )
        metric = AnswerRelevancyMetric(    # Step 2: Define the metric
            threshold=self.THRESHOLD,
            verbose_mode=True              # Print scoring details in output
        )
        test_case = LLMTestCase(           # Step 3: Package inputs/outputs
            input=result["input"],         #   what we asked
            actual_output=result["output"] #   what the LLM said
        )
        assert_test(test_case, [metric])   # Step 4: Score + assert

    def test_capital_city_question(self, pipeline):
        self._run(
            pipeline,
            question="What is the capital of France?",
            context="France is in Western Europe. Its capital is Paris."
        )
```

**The `_run` helper:** Avoids repeating the same 4-step pattern in every test. This is the **DRY principle** (Don't Repeat Yourself).

---

### Parametrized Dataset Tests

```python
@pytest.mark.parametrize("tc_id", ["tc_001", "tc_002", "tc_004"])
def test_dataset_factual_cases(self, pipeline, test_dataset, tc_id):
    tc = next(t for t in test_dataset if t["id"] == tc_id)
    self._run(pipeline, question=tc["question"], context=tc.get("context", ""))
```

**What `@pytest.mark.parametrize` does:** Runs the same test function 3 times, once for each `tc_id`. This is equivalent to writing 3 separate test functions but much cleaner.

**Why drive from a dataset:** Separates test logic from test data. You can add 50 new test cases by editing the JSON file without touching any Python code. This is a key SDET best practice.

---

### `test_faithfulness.py` — The Key Difference

```python
test_case = LLMTestCase(
    input=result["input"],
    actual_output=result["output"],
    retrieval_context=[context],      # ← THIS is what makes it a faithfulness test
)
```

Faithfulness requires `retrieval_context` — the documents the model was given. Without it, there's nothing to check faithfulness against.

```python
def test_world_cup_out_of_context(self, pipeline):
    """Model should admit it doesn't know rather than hallucinate."""
    self._run(
        pipeline,
        question="Who won the 2050 World Cup?",
        context="The 2022 FIFA World Cup was held in Qatar. Argentina won...",
    )
```

This test checks that when asked about something NOT in the context, the model says "I don't know" rather than inventing an answer. A well-prompted model (with our system prompt) should pass this.

---

## 9. The Test Dataset

**File:** `evals/datasets/qa_test_cases.json`

The dataset separates **test data** from **test logic**. This is a fundamental SDET principle.

```json
{
  "id": "tc_001",            // Unique ID for traceability
  "category": "factual",     // Category for filtering/grouping
  "question": "What is the capital of France?",
  "context": "France is in Western Europe. Its capital city is Paris.",
  "expected_output": "The capital of France is Paris.",  // Reference only
  "tags": ["geography", "factual"]   // For selective test runs
}
```

### Test Categories Included

| Category | Purpose |
|----------|---------|
| `factual` | Verify accurate retrieval from context |
| `reasoning` | Check logical/math questions (no context) |
| `hallucination_check` | Verify model doesn't invent numbers or facts |
| `out_of_context` | Verify model admits ignorance rather than hallucinating |
| `summarization` | Check condensed, accurate summaries |

### `expected_output` — Why It's Not Used in Assertions

You'll notice `expected_output` is in the dataset but we don't do:
```python
assert result["output"] == tc["expected_output"]
```
Because LLM outputs are non-deterministic. Instead, it serves as:
- **Human reference** — for developers to understand what a good answer looks like
- **Future use** — some metrics like `GEval` can use it as a scoring reference
- **Documentation** — makes the dataset self-explanatory

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
