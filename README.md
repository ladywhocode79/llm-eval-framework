# LLM Eval Framework

A basic LLM testing framework built with [deepeval](https://github.com/confident-ai/deepeval) and the Anthropic Claude API.

## What's Inside

```
llm-eval-framework/
├── app/
│   ├── llm_client.py        # Claude API wrapper
│   ├── qa_pipeline.py       # Q&A pipeline (the LLM app being tested)
│   ├── local_judge.py       # Ollama judge wrapper
│   └── judge_factory.py     # Resolves the judge model at runtime (ollama / gemini / anthropic)
├── evals/
│   ├── datasets/
│   │   ├── qa_test_cases.json   # Reusable test case dataset
│   │   ├── golden_set.json      # Meal-planner agent scenarios (tool calls + expected output)
│   │   └── human_annotated_sets.json  # Human-labelled gold set for judge calibration
│   ├── metrics/
│   │   └── custom_metrics.py    # Deterministic custom metrics
│   └── tests/
│       ├── test_answer_relevancy.py   # LLM-as-judge: relevancy
│       ├── test_faithfulness.py       # LLM-as-judge: hallucination
│       ├── test_custom_metrics.py     # Deterministic: keyword, length, numbers
│       ├── test_evals.py              # Meal-planner agent: tool-call schema + judged faithfulness/safety/relevancy
│       ├── test_judge_calibration.py  # Judge vs. human agreement (Cohen's Kappa)
│       └── test_model_benchmark.py    # Haiku 4.5 vs Sonnet 5.5 judge: κ, latency, cost
├── docs/                    # Split testing guide + case study (start at LLM_TESTING_GUIDE.md)
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
| `test_evals.py` | Tool-call schema (Pydantic) | Deterministic |
| `test_evals.py` | FaithfulnessMetric | LLM-as-judge (Claude) |
| `test_evals.py` | AnswerRelevancyMetric | LLM-as-judge (Claude) |
| `test_evals.py` | allergen_safety_metric (GEval) | LLM-as-judge (Claude) |
| `test_judge_calibration.py` | Cohen's Kappa vs. human labels | Judge calibration |
| `test_model_benchmark.py` | κ, latency, cost across judge models | Benchmark |

For explanations, challenges and learnings, see the guide: [LLM_TESTING_GUIDE.md](LLM_TESTING_GUIDE.md).

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

## Meal-Planner Agent Evals (`test_evals.py`)

`test_evals.py` tests a meal-planner agent scenario against `evals/datasets/golden_set.json`, a set of persona-driven scenarios (e.g. dietary restrictions, allergies) each with an expected tool call and expected output. Each scenario runs:

- **Layer A — Tool-call schema (deterministic):** validates `expected_tool_call.args` against the `FetchRecipesArgs` Pydantic schema.
- **Layer B — LLM-as-judge:** `FaithfulnessMetric` always; `allergen_safety_metric` (a custom `GEval` metric) only for scenarios that declare an allergen; `AnswerRelevancyMetric` for every scenario except ones marked `"expects_refusal": true` (relevancy penalizes a correct safety refusal for lacking "actionable suggestions").

**Judge model is resolved at runtime, not hardcoded** — see `app/judge_factory.py`. `EVAL_JUDGE_BACKEND` in `.env` controls it (falls back to the shared `JUDGE_BACKEND` if unset):

| Value | Behavior |
|-------|----------|
| `auto` (default) | Prefer local Ollama if it's running and the model is pulled; else fall back to `GEMINI_API_KEY`, then `ANTHROPIC_API_KEY` |
| `ollama` | Force local Ollama (`OLLAMA_MODEL`) |
| `gemini` | Force Gemini (requires `GEMINI_API_KEY`) |
| `anthropic` | Force Claude (requires `ANTHROPIC_API_KEY`) |

In practice, prefer local for simple checks, but verify it against your actual rubric before trusting it on safety-critical ones — testing showed llama3.2 hallucinating facts about the retrieval context on the allergen-safety rubric here, so this project's `.env` pins `EVAL_JUDGE_BACKEND=anthropic` for this file while the other eval files stay on the local Ollama judge (see [case study 6](docs/case-study/06-model-agnostic-judge.md)).

Add new scenarios by appending an object to `golden_set.json`:
```json
{
  "scenario_id": "MP_XXX_003",
  "persona": "Short description of the user",
  "input": "User's request to the meal planner",
  "retrieved_context": ["Recipe_301: ...", "Recipe_302: ..."],
  "expected_tool_call": {
    "name": "fetch_recipes",
    "args": { "meal_type": "dinner" }
  },
  "actual_output": "The agent's actual response",
  "expected_output": "The ideal response"
}
```

## Judge Calibration (`test_judge_calibration.py`)

LLM judges are only useful if they agree with a human. This test measures that: it scores a small human-labelled set with the **same judge, rubric and metrics as `test_evals.py`** and asserts **Cohen's Kappa ≥ 0.80** between the judge's pass/fail and the human label.

```bash
# Run only calibration (makes live judge calls — costs tokens on cloud judges)
pytest -m calibration -v -s

# Run the calibration test against a specific judge
EVAL_JUDGE_BACKEND=anthropic pytest -m calibration -v -s
```

How it works:
- **Data:** `evals/datasets/human_annotated_sets.json`, loaded by `load_calibration_dataset()` in `test_evals.py`.
- **Metrics per case:** `FaithfulnessMetric` + `allergen_safety_metric`, plus `AnswerRelevancyMetric` unless the case has `"expects_refusal": true`. A case is judge-pass (1) only if **all** metrics pass, otherwise 0.
- **Result:** if κ < 0.80 the failure message lists the disagreeing `scenario_id`s. Read each case's `reasoning` to decide whether the judge or the label is wrong.
- **Importing `test_evals.py` builds the judge**, so the chosen backend must be reachable (Ollama running, or an API key set).
- **Not part of `-m eval`.** It carries its own `calibration` marker; plain `pytest` (which uses `testpaths`) will run it too.

Gold-set entry format:
```json
{
  "scenario_id": "CALIB_011",
  "persona": "Short description of the user",
  "input": "User's request",
  "actual_output": "The agent response to judge",
  "retrieved_context": ["Recipe_...: ..."],
  "human_label": 1,
  "expects_refusal": false,
  "reasoning": "Why a human labelled it 1 (acceptable) or 0 (unacceptable)"
}
```

Guidelines for adding cases:
- Label `1` = response a human would accept, `0` = one they would reject. Always fill in `reasoning`.
- Set `expects_refusal: true` when refusing is the correct answer (otherwise relevancy will fail it).
- Keep the set roughly balanced and cover every failure type your metrics should catch. With 10 cases, one disagreement gives κ = 0.80 and two give 0.60, so grow the set before drawing conclusions from small κ differences.
- Re-run calibration whenever the judge model, rubric, threshold or backend changes.

Details and lessons learned: [docs/case-study/07-judge-calibration.md](docs/case-study/07-judge-calibration.md).

## Judge Model Benchmark (`test_model_benchmark.py`)

Compares two Claude judges (`claude-haiku-4-5-20251001`, `claude-sonnet-5-5`) on the same calibration set and the same `allergen_safety_metric` rubric, and prints a table of Cohen's Kappa, p50/p95 latency, total time and estimated cost.

```bash
pytest -m benchmark -v -s     # needs ANTHROPIC_API_KEY; makes live calls to both models
```

Output: the table is printed and also saved to `reports/model_benchmark.md` and `reports/model_benchmark.json` (stdout is not captured in `report.html`).

Notes:
- First observed run: Haiku 4.5 scored κ = 0.60, so the test failed its 0.80 gate; Sonnet 5.5 numbers weren't captured in that run (the report-saving fix above was added afterwards). Re-run to get both.
- Cost uses fixed token estimates (350 in / 150 out per case) and an **assumed** Sonnet 5.5 price — verify pricing in the file.
- `claude-sonnet-5-5` rejects `temperature`; `ClaudeLLM` retries without it, so Sonnet is not pinned to `temperature=0` and results can vary run to run. Re-run before drawing conclusions.
- It scores only the safety metric, so `CALIB_007` (off-topic) is likely a miss for both models.

Details: [docs/case-study/08-model-benchmark.md](docs/case-study/08-model-benchmark.md).

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'ollama'` | Package not installed | `pip install -r requirements.txt` |
| `model 'llama3.2' not found (status code: 404)` | Model not pulled | `ollama pull llama3.2` |
| `Cannot connect to Ollama server` | Server not running | `ollama serve` (in a new terminal) |
| `ANTHROPIC_API_KEY` not set | Missing env config | Copy `.env.example` → `.env` and add your key |
| Calibration fails with κ < 0.80 | Judge disagrees with human labels | Check the listed `scenario_id`s and their `reasoning`; fix the label, the rubric, or switch judge |
| `temperature is deprecated for this model` (400) | Newer Claude model rejects `temperature` | Handled automatically by `ClaudeLLM` (retries without it); update `app/judge_factory.py` if you see it elsewhere |
| Calibration errors at import | Judge backend unreachable (imports `test_evals.py`) | Start Ollama or set the API key for `EVAL_JUDGE_BACKEND` |
| Any of the above unclear | — | Run `python scripts/check_setup.py` for a guided diagnosis |

## Running Evals

```bash
# Run all eval tests
pytest -m eval -v

# Run a specific test file
pytest evals/tests/test_answer_relevancy.py -v
pytest evals/tests/test_faithfulness.py -v
pytest evals/tests/test_custom_metrics.py -v
pytest evals/tests/test_evals.py -v
pytest -m calibration -v -s    # judge calibration

# Run with detailed deepeval output
pytest -m eval -v -s

# Run only dataset-driven parametrized tests
pytest -m eval -k "dataset" -v
```

## How to Execute Test Cases

### Prerequisites
Make sure setup is complete and your virtual environment is active:
```bash
cd /Applications/myapps/llm-eval-framework
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
| Meal-planner agent scenarios (golden set) | `pytest evals/tests/test_evals.py -v` |
| Judge calibration (Cohen's Kappa) | `pytest -m calibration -v -s` |
| Judge model benchmark | `pytest -m benchmark -v -s` |

---

### Run a Single Test

```bash
# Pattern: pytest <file>::<Class>::<method> -v
pytest evals/tests/test_answer_relevancy.py::TestAnswerRelevancy::test_capital_city_question -v
pytest evals/tests/test_faithfulness.py::TestFaithfulness::test_boiling_point_grounded -v
pytest evals/tests/test_custom_metrics.py::TestCustomMetrics::test_keyword_capital_paris -v

# Meal-planner scenarios are parametrized by scenario_id from golden_set.json
pytest "evals/tests/test_evals.py::test_meal_planner_scenario[MP_VAL_001]" -v
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

## Documentation

The full guide is split by topic under [docs/](docs/), indexed by [LLM_TESTING_GUIDE.md](LLM_TESTING_GUIDE.md): fundamentals, metrics, local judge, running tests, interview points, and a [case study](docs/case-study/README.md) of the challenges hit (including judge calibration).

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
