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

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
