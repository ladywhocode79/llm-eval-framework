# LLM Eval Framework

A basic LLM testing framework built with [deepeval](https://github.com/confident-ai/deepeval) and the Anthropic Claude API.

## What's Inside

```
llm-eval-framework/
├── app/
│   ├── llm_client.py        # Claude API wrapper
│   └── qa_pipeline.py       # Q&A pipeline (the LLM app being tested)
├── evals/
│   ├── datasets/
│   │   └── qa_test_cases.json   # Reusable test case dataset
│   ├── metrics/
│   │   └── custom_metrics.py    # Deterministic custom metrics
│   └── tests/
│       ├── test_answer_relevancy.py   # LLM-as-judge: relevancy
│       ├── test_faithfulness.py       # LLM-as-judge: hallucination
│       └── test_custom_metrics.py     # Deterministic: keyword, length, numbers
├── conftest.py              # Shared pytest fixtures
├── pytest.ini
└── requirements.txt
```

## Eval Types

| File | Metric | Type |
|------|--------|------|
| `test_answer_relevancy.py` | AnswerRelevancyMetric | LLM-as-judge |
| `test_faithfulness.py` | FaithfulnessMetric | LLM-as-judge |
| `test_custom_metrics.py` | KeywordPresentMetric | Deterministic |
| `test_custom_metrics.py` | OutputLengthMetric | Deterministic |
| `test_custom_metrics.py` | NoHallucinatedNumberMetric | Deterministic |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Running Evals

```bash
# Run all eval tests
pytest -m eval -v

# Run a specific test file
pytest evals/tests/test_answer_relevancy.py -v
pytest evals/tests/test_faithfulness.py -v
pytest evals/tests/test_custom_metrics.py -v

# Run with detailed deepeval output
pytest -m eval -v -s

# Run only dataset-driven parametrized tests
pytest -m eval -k "dataset" -v
```

## Adding New Test Cases

**Option 1 — Add to the dataset** (`evals/datasets/qa_test_cases.json`):
```json
{
  "id": "tc_007",
  "category": "factual",
  "question": "Your question here",
  "context": "Supporting context...",
  "expected_output": "Expected answer",
  "tags": ["tag1"]
}
```

**Option 2 — Write a new test directly:**
```python
def test_my_custom_eval(self, pipeline):
    result = pipeline.answer(question="...", context="...")
    metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
    assert_test(test_case, [metric])
```

**Option 3 — Add a custom metric** (`evals/metrics/custom_metrics.py`):
Subclass `BaseMetric` and implement `measure()`, `is_successful()`, and `name`.

## Architecture

```
Test Case (question + context)
        │
        ▼
  QAPipeline.answer()          ← the LLM app
        │
        ▼
  actual_output (LLM response)
        │
        ▼
  deepeval Metric(s)           ← the evaluator
        │
        ▼
  assert_test()                ← pass / fail
```
