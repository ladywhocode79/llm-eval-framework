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

# 3. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (required) and judge settings (see below)
```

## Local Judge Setup (Ollama — Recommended)

The LLM-as-judge evaluator runs **locally via Ollama** by default — no OpenAI key or API cost needed.

### Step 1 — Install Ollama

```bash
# Mac
brew install ollama

# Or download from https://ollama.com
```

### Step 2 — Pull a judge model (one-time, ~2 GB)

```bash
ollama pull llama3.2        # recommended — fast, ~2 GB
# ollama pull llama3.2:1b   # lightest — ~1 GB, lower quality
# ollama pull mistral        # best quality — ~4 GB, slower
```

### Step 3 — Start the Ollama server

```bash
# Open a new terminal and keep this running during your test session
ollama serve
```

> Ollama must be running before executing any LLM-as-judge tests.

### Step 4 — Verify your setup

Run the setup checker before your first test run:

```bash
python scripts/check_setup.py
```

Example output (all passing):
```
=== LLM Eval Framework — Setup Check ===

  [PASS] Python version — 3.12.0
  [PASS] anthropic package
  [PASS] deepeval package
  [PASS] ollama package
  [PASS] ANTHROPIC_API_KEY — set (sk-ant-a...)
  [INFO] JUDGE_BACKEND  — ollama
  [INFO] OLLAMA_MODEL   — llama3.2
  [PASS] Ollama server  — running
  [PASS] Ollama model   — 'llama3.2' is available
  [PASS] Test dataset file — found

=== All checks passed. You're ready to run tests! ===
```

### Switch judge backend

In your `.env` file:

```bash
# Use local Ollama judge (default)
JUDGE_BACKEND=ollama
OLLAMA_MODEL=llama3.2

# Use OpenAI judge (requires OPENAI_API_KEY)
JUDGE_BACKEND=openai
OPENAI_API_KEY=your-key-here
```

### Judge Model Comparison

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| `llama3.2` | ~2 GB | Fast | Good | `ollama pull llama3.2` |
| `llama3.2:1b` | ~1 GB | Fastest | Lower | `ollama pull llama3.2:1b` |
| `mistral` | ~4 GB | Slow | Best | `ollama pull mistral` |
| `phi3:mini` | ~2.3 GB | Medium | Good | `ollama pull phi3:mini` |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'ollama'` | Package not installed | `pip install -r requirements.txt` |
| `model 'llama3.2' not found (status code: 404)` | Model not pulled | `ollama pull llama3.2` |
| `Cannot connect to Ollama server` | Server not running | `ollama serve` (in a new terminal) |
| `ANTHROPIC_API_KEY` not set | Missing env config | Copy `.env.example` → `.env` and add your key |
| Any of the above unclear | — | Run `python scripts/check_setup.py` for a guided diagnosis |

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

## How to Execute Test Cases

### Prerequisites
Make sure setup is complete and your virtual environment is active:
```bash
cd "/Applications/my apps/llm-eval-framework"
source venv/bin/activate
```

---

### Run All Tests
```bash
pytest -m eval -v
```

---

### Run by Test File

| Goal | Command |
|------|---------|
| Answer relevancy tests | `pytest evals/tests/test_answer_relevancy.py -v` |
| Faithfulness / hallucination tests | `pytest evals/tests/test_faithfulness.py -v` |
| Custom deterministic metric tests | `pytest evals/tests/test_custom_metrics.py -v` |

---

### Run a Single Test

```bash
# Pattern: pytest <file>::<Class>::<method> -v
pytest evals/tests/test_answer_relevancy.py::TestAnswerRelevancy::test_capital_city_question -v
pytest evals/tests/test_faithfulness.py::TestFaithfulness::test_boiling_point_grounded -v
pytest evals/tests/test_custom_metrics.py::TestCustomMetrics::test_keyword_capital_paris -v
```

---

### Run by Category / Tag

```bash
# Run only dataset-driven parametrized tests
pytest -m eval -k "dataset" -v

# Run only tests related to faithfulness
pytest -m eval -k "faithful" -v

# Run only keyword and length metric tests
pytest -m eval -k "keyword or length" -v

# Exclude LLM-as-judge tests (run only deterministic/cheap tests)
pytest evals/tests/test_custom_metrics.py -v
```

---

### Run with Detailed Output

```bash
# Show deepeval scores and reasoning in the terminal
pytest -m eval -v -s

# Stop after the first failure
pytest -m eval -v -x

# Show 5 slowest tests
pytest -m eval -v --durations=5
```

---

### Expected Output

**Passing test:**
```
PASSED evals/tests/test_answer_relevancy.py::TestAnswerRelevancy::test_capital_city_question
```

**Failing test:**
```
FAILED evals/tests/test_faithfulness.py::TestFaithfulness::test_world_cup_out_of_context

AssertionError: FaithfulnessMetric (score: 0.3, threshold: 0.7)
Reason: Output contains claims not supported by the retrieval context.
```

**Full run summary:**
```
============== 15 passed, 1 failed in 42.3s ==============
```

---

### HTML Report

An HTML report is generated automatically after every run at:
```
reports/report.html
```

Open it in any browser:
```bash
open reports/report.html          # Mac
xdg-open reports/report.html      # Linux
start reports/report.html         # Windows
```

The report is self-contained (single `.html` file — no extra assets needed) and includes:
- Pass/fail status per test
- Error messages and deepeval failure reasons
- Test duration
- Environment metadata

To save a timestamped copy instead of overwriting each run:
```bash
pytest -m eval -v --html=reports/report_$(date +%Y%m%d_%H%M%S).html --self-contained-html
```

---

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
