# LLM Testing Framework — Complete Guide
### For SDETs New to LLM Evaluation

---

## Table of Contents

1. [What is LLM Testing?](#1-what-is-llm-testing)
2. [Why LLM Testing is Different from Traditional API Testing](#2-why-llm-testing-is-different-from-traditional-api-testing)
3. [Key Concepts You Must Know](#3-key-concepts-you-must-know)
4. [Framework Architecture — The Big Picture](#4-framework-architecture--the-big-picture)
5. [The LLM App We Are Testing](#5-the-llm-app-we-are-testing)
6. [The Eval Framework](#6-the-eval-framework)
7. [Types of Metrics Explained](#7-types-of-metrics-explained)
8. [Test Files — Line by Line Walkthrough](#8-test-files--line-by-line-walkthrough)
9. [The Test Dataset](#9-the-test-dataset)
10. [How to Run the Tests](#10-how-to-run-the-tests)
11. [Reading Test Results](#11-reading-test-results)
12. [Interview Talking Points](#12-interview-talking-points)
13. [Glossary](#13-glossary)

---

## 1. What is LLM Testing?

An **LLM (Large Language Model)** is an AI system (like ChatGPT or Claude) that generates text responses to natural language inputs.

**LLM Testing** is the practice of systematically checking whether an LLM-powered application behaves correctly, safely, and reliably. It answers questions like:

- Does the AI answer the question that was actually asked?
- Does the AI make up facts that aren't true (hallucination)?
- Does the AI stay within the bounds of the provided information?
- Is the response too short, too long, or missing key information?
- Is the response harmful, biased, or toxic?

### Why This Matters for SDETs

As LLM-powered features appear in more products (chatbots, search assistants, code helpers, customer support bots), the **SDET's job now includes evaluating AI quality** — not just API status codes and response schemas. This is a growing and highly valued skill.

---

## 2. Why LLM Testing is Different from Traditional API Testing

This is the most important concept to understand before an interview.

| Aspect | Traditional API Testing | LLM Testing |
|--------|------------------------|-------------|
| **Expected output** | Exact and deterministic (`"status": "OK"`) | Non-deterministic — varies every run |
| **Pass/fail criteria** | Exact match or schema validation | Semantic quality scores (0.0–1.0) |
| **Test oracle** | You know the correct answer exactly | Correct answer is subjective or approximate |
| **Failure modes** | HTTP errors, wrong fields, wrong values | Hallucination, irrelevance, bias, toxicity |
| **Evaluation method** | `assert response.status_code == 200` | LLM-as-judge, embedding similarity, keyword checks |
| **Repeatability** | Same input → same output every time | Same input → slightly different output each time |
| **Speed** | Milliseconds | Seconds (each test calls the AI API) |
| **Cost** | Free (local logic) | Costs API tokens per test run |

### The Core Challenge

You **cannot** write:
```python
assert response == "The capital of France is Paris."
```
Because the model might say:
- "Paris is the capital of France."
- "France's capital city is Paris."
- "The answer is Paris."

All three are **correct**, but none match exactly. LLM eval uses **semantic metrics** to measure quality instead of exact matches.

---

## 3. Key Concepts You Must Know

### 3.1 Prompt
The text input you send to the LLM. In our framework, a prompt is built from a **question** + optional **context**.

```
Context: France is a country in Western Europe. Its capital is Paris.
Question: What is the capital of France?
```

### 3.2 Context (Retrieval Context)
Supporting information given to the model to answer from. This simulates a **RAG (Retrieval-Augmented Generation)** pattern — where relevant documents are fetched and passed alongside the question.

**With context:** Model is expected to answer using ONLY the provided text.
**Without context:** Model uses its internal training knowledge.

### 3.3 Hallucination
When an LLM **confidently states something false** that is not supported by the provided context or factual reality.

Example:
- Context says: "Water boils at 100°C."
- Model says: "Water boils at 95°C." ← hallucination

Hallucination testing is one of the most critical aspects of LLM evaluation.

### 3.4 LLM-as-Judge
Using a **separate LLM call** to evaluate the quality of another LLM's response. Instead of hardcoded rules, you ask an AI to score the output on a 0–1 scale.

```
Evaluator LLM prompt:
"Given this question: [question]
And this answer: [answer]
Rate how relevant the answer is to the question. Score: 0.0 to 1.0"
```

This is how `AnswerRelevancyMetric` and `FaithfulnessMetric` work in our framework.

### 3.5 Threshold
The minimum acceptable score (0.0–1.0) for a metric to be considered passing.

```python
AnswerRelevancyMetric(threshold=0.7)
# A score of 0.7 or above = PASS
# A score below 0.7      = FAIL
```

### 3.6 Test Case (in deepeval)
A structured object containing:
- `input` — the question asked
- `actual_output` — what the LLM produced
- `expected_output` — (optional) what we expected
- `retrieval_context` — the context documents used

### 3.7 RAG (Retrieval-Augmented Generation)
A common LLM app pattern:
1. User asks a question
2. System retrieves relevant documents from a database
3. Documents + question are sent to the LLM
4. LLM answers using the retrieved documents

Our `QAPipeline` simulates this — we manually pass `context` instead of fetching from a database.

---

## 4. Framework Architecture — The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM EVAL FRAMEWORK                         │
│                                                                 │
│  ┌──────────────────────┐      ┌───────────────────────────┐   │
│  │   THE APP (app/)     │      │  THE EVALS (evals/)       │   │
│  │                      │      │                           │   │
│  │  llm_client.py       │      │  datasets/                │   │
│  │  (Claude API calls)  │      │  (test input data)        │   │
│  │         ↓            │      │         ↓                 │   │
│  │  qa_pipeline.py      │◄─────│  tests/                   │   │
│  │  (builds prompts,    │      │  (call pipeline, get       │   │
│  │   returns answers)   │─────►│   output, run metrics)    │   │
│  └──────────────────────┘      │         ↓                 │   │
│                                │  metrics/                 │   │
│                                │  (score the output)       │   │
│                                │         ↓                 │   │
│                                │  PASS / FAIL              │   │
│                                └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow — Step by Step

```
1. Test starts
        │
2. Load test input (question + context) from dataset or hardcoded
        │
3. Call QAPipeline.answer(question, context)
        │
4. qa_pipeline builds the prompt:
   "Context: {context}\nQuestion: {question}"
        │
5. LLMClient sends prompt to Claude API
        │
6. Claude returns a text response
        │
7. Build a deepeval LLMTestCase with input + actual_output
        │
8. Run metric(s) against the test case
   - LLM-as-judge: sends another API call to score quality
   - Deterministic: runs Python logic to score
        │
9. Compare score vs threshold → PASS or FAIL
        │
10. pytest reports result
```

---

## 5. The LLM App We Are Testing

We built a simple Q&A app that simulates a real-world AI assistant.

### `app/llm_client.py` — The API Wrapper

```python
class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        return response.content[0].text
```

**What it does:** Takes a prompt string → sends it to Claude → returns the text response.

**Why we wrap it:** Abstraction. If we swap Claude for GPT-4 or Gemini later, we only change this one file. All tests stay the same. This is the **Adapter Pattern**.

---

### `app/qa_pipeline.py` — The Application Logic

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions accurately.
When context is provided, base your answer strictly on that context.
If the context does not contain enough information, say so clearly.
Do not make up facts."""

class QAPipeline:
    def answer(self, question: str, context: str = "") -> dict:
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = f"Question: {question}"

        output = self.client.complete(prompt=prompt, system=SYSTEM_PROMPT)
        return {"input": question, "context": context, "output": output}
```

**What it does:**
- Takes a question and optional context
- Builds a structured prompt
- Returns a dict with `input`, `context`, and `output`

**Why return a dict:** Tests need all three values — the input to evaluate relevancy, the context to check faithfulness, and the output to score. Packaging them together keeps test code clean.

**The System Prompt role:** Acts as persistent instructions that frame every conversation. It tells the model to: be accurate, stay in context, and never hallucinate. This is your first line of defense against bad outputs.

---

## 6. The Eval Framework

### Why deepeval?

`deepeval` is a Python library specifically built for LLM evaluation. It:
- Integrates natively with `pytest` (no new tooling to learn)
- Provides pre-built metrics (relevancy, faithfulness, hallucination, toxicity)
- Lets you write custom metrics
- Produces detailed failure reasons (not just pass/fail)
- Supports both LLM-as-judge and deterministic metrics

### `conftest.py` — Shared Fixtures

```python
@pytest.fixture(scope="session")
def pipeline():
    return QAPipeline()  # One Claude client for all tests

@pytest.fixture(scope="session")
def test_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)  # Load test cases once
```

**`scope="session"`** means the fixture is created once and shared across all tests. This avoids creating a new API client for every single test — saving time and avoiding rate limits.

---

## 7. Types of Metrics Explained

### 7.1 AnswerRelevancyMetric (LLM-as-Judge)

**File:** `evals/tests/test_answer_relevancy.py`

**Question it answers:** "Does the model's response actually address what was asked?"

**How it works internally:**
1. Takes `input` (question) and `actual_output` (LLM response)
2. Sends BOTH to an evaluator LLM (also Claude)
3. The evaluator scores: "How relevant is this answer to this question?" → 0.0 to 1.0
4. Score ≥ threshold (0.7) = PASS

**Real example:**
```
Input:    "What is the capital of France?"
Output:   "Paris is a beautiful city with the Eiffel Tower."

Score: 0.5 — FAIL (mentions Paris but doesn't directly answer)
```
```
Input:    "What is the capital of France?"
Output:   "The capital of France is Paris."

Score: 1.0 — PASS (directly and completely answers)
```

---

### 7.2 FaithfulnessMetric (LLM-as-Judge)

**File:** `evals/tests/test_faithfulness.py`

**Question it answers:** "Does the model's response stick to the provided context, or does it make things up?"

**How it works internally:**
1. Takes `actual_output` and `retrieval_context` (the documents)
2. Evaluator LLM checks: are the claims in the output supported by the context?
3. Score = (supported claims) / (total claims in output)

**Real example:**
```
Context: "Water boils at 100°C at 1 atm."
Output:  "Water boils at 100°C."

Score: 1.0 — PASS (all claims grounded in context)
```
```
Context: "Water boils at 100°C at 1 atm."
Output:  "Water boils at 100°C. It also freezes at -5°C."

Score: 0.5 — FAIL (freezing point not in context = hallucination)
```

**This is the hallucination detection metric.** Critical for any AI app that operates over documents (legal, medical, financial).

---

### 7.3 KeywordPresentMetric (Deterministic Custom)

**File:** `evals/metrics/custom_metrics.py`

**Question it answers:** "Are required keywords present in the output?"

**How it works:**
```python
found = [kw for kw in self.keywords if kw in output.lower()]
score = len(found) / len(self.keywords)
```

No LLM call needed — pure Python string matching. Fast and cheap.

**When to use:** When you know specific terms MUST appear. Example: a medical disclaimer must always contain "consult a doctor."

---

### 7.4 OutputLengthMetric (Deterministic Custom)

**Question it answers:** "Is the response within an acceptable length range?"

**How it works:**
```python
word_count = len(output.split())
score = 1.0 if min_words <= word_count <= max_words else 0.0
```

**When to use:**
- A "one-sentence summary" should not be 500 words
- A detailed report should not be 3 words
- Prevents lazy ("I don't know") or runaway responses

---

### 7.5 NoHallucinatedNumberMetric (Deterministic Custom)

**Question it answers:** "Did the model invent any numerical values not present in the context?"

**How it works:**
```python
output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", output))
context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context))
hallucinated = output_numbers - context_numbers
```

**Why this matters:** Numbers (prices, dates, statistics, dosages) are the most dangerous things an LLM can hallucinate because they sound authoritative and are easy to miss in a review.

---

### Metric Comparison Summary

| Metric | Type | Uses LLM? | Cost | Speed | Best For |
|--------|------|-----------|------|-------|----------|
| AnswerRelevancyMetric | LLM-as-judge | Yes | High | Slow | Relevance quality |
| FaithfulnessMetric | LLM-as-judge | Yes | High | Slow | Hallucination detection |
| KeywordPresentMetric | Deterministic | No | Free | Fast | Required terms |
| OutputLengthMetric | Deterministic | No | Free | Fast | Length guardrails |
| NoHallucinatedNumberMetric | Deterministic | No | Free | Fast | Numeric accuracy |

**Strategy:** Run deterministic metrics in every CI pipeline (fast, free). Run LLM-as-judge metrics in scheduled eval runs or before releases (slower, costs tokens).

---

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

### "How does the framework scale?"

> "The current framework can scale in a few ways: adding more test cases to the JSON dataset without code changes, adding new metrics by subclassing BaseMetric, integrating with CI/CD to run on every deployment, connecting to a real vector database for genuine RAG testing, and adding more metrics like toxicity or bias detection. The architecture keeps the app and evals separate, so we can point the same eval framework at different LLM backends."

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **LLM** | Large Language Model — an AI trained on text to generate language (e.g., Claude, GPT-4) |
| **Eval / Evaluation** | Measuring the quality of an LLM's output using defined metrics |
| **Hallucination** | When an LLM generates false information with apparent confidence |
| **RAG** | Retrieval-Augmented Generation — fetching documents and including them in the prompt |
| **Context** | Supporting documents given to the LLM to base its answer on |
| **Prompt** | The full text input sent to the LLM |
| **System Prompt** | Persistent instructions that shape the model's behavior across a conversation |
| **LLM-as-Judge** | Using an LLM to score/evaluate another LLM's output |
| **Threshold** | Minimum score (0.0–1.0) for a metric to be considered passing |
| **Deterministic Metric** | A metric computed by pure code logic, no AI involved |
| **Non-deterministic** | Output varies between runs even with identical input |
| **deepeval** | Python library for LLM evaluation, integrates with pytest |
| **Faithfulness** | Whether the model's claims are supported by the provided context |
| **Relevancy** | Whether the model's response addresses what was actually asked |
| **Test Case** | A structured unit: input + context + actual output + optional expected output |
| **Fixture** | A pytest mechanism for sharing setup code across multiple tests |
| **Parametrize** | A pytest feature to run one test function with multiple input sets |
| **Token** | The unit of text that LLMs process (roughly 0.75 words); API cost is per token |
| **SDET** | Software Development Engineer in Test — engineers who build test frameworks and automation |

---

*Framework built with: Python · deepeval · Anthropic Claude SDK · pytest*
