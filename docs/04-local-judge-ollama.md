## 7.6 Local LLM Judge — Ollama (Cost-Free Evaluation)

### The Problem with Cloud-Based Judges

By default, deepeval uses OpenAI (GPT-4) to judge outputs. Every metric evaluation makes an API call that costs money. In a test suite with 15 tests using LLM-as-judge, you make 15+ extra API calls just for evaluation — on top of the calls the app already makes.

| Approach | Cost per test run | Internet required | Consistency |
|----------|-----------------|-------------------|-------------|
| OpenAI judge (default) | ~$0.10–$0.50 | Yes | High |
| Ollama local judge | $0.00 | No (after setup) | High |

### What is Ollama?

Ollama is a tool that lets you run open-source LLMs (like Llama, Mistral, Phi) locally on your machine. It exposes them via a simple REST API so any code that talks to an LLM can use it.

```
Without Ollama:                    With Ollama:
  Your tests                         Your tests
      │                                  │
      ▼                                  ▼
  OpenAI API (internet)           Ollama (localhost:11434)
  cost per call                         │
                                  Local model (free)
```

### How `app/local_judge.py` Works

```python
from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

class OllamaJudge(DeepEvalBaseLLM):

    def load_model(self):
        return self.model_name           # tells deepeval which model

    def generate(self, prompt: str) -> Tuple[str, float]:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content, 0.0   # (text, cost=0 — it's free)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"
```

`DeepEvalBaseLLM` is deepeval's abstract base class for plugging in any LLM. By subclassing it and implementing `generate()`, our local Ollama model becomes a first-class deepeval judge — no other changes needed.

### How the `judge` Fixture Controls Everything

In `conftest.py`:
```python
@pytest.fixture(scope="session")
def judge():
    backend = os.environ.get("JUDGE_BACKEND", "ollama")   # default: local
    if backend == "ollama":
        return OllamaJudge(model=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    return None   # None → deepeval uses OpenAI
```

The `judge` fixture is injected into every LLM-as-judge test:
```python
def test_capital_city(self, pipeline, judge):
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)  # ← plugged in here
```

Passing `model=judge` overrides deepeval's default. Passing `model=None` falls back to OpenAI.

### Choosing a Judge Model

| Model | Pull command | Size | Speed | Best for |
|-------|-------------|------|-------|----------|
| `llama3.2` | `ollama pull llama3.2` | ~2 GB | Fast | General use — recommended |
| `llama3.2:1b` | `ollama pull llama3.2:1b` | ~1 GB | Fastest | Quick feedback, CI pipelines |
| `mistral` | `ollama pull mistral` | ~4 GB | Slow | Highest accuracy |
| `phi3:mini` | `ollama pull phi3:mini` | ~2.3 GB | Medium | Good balance |

**Rule of thumb:** Use `llama3.2` for local development. If you need a quality gate before a release, switch to `mistral` or even OpenAI by setting `JUDGE_BACKEND=openai`.

### Switching Backends via `.env`

```bash
# Local (default) — free
JUDGE_BACKEND=ollama
OLLAMA_MODEL=llama3.2

# Cloud — accurate but costs money
JUDGE_BACKEND=openai
OPENAI_API_KEY=sk-...
```

No code changes needed — just change the env var and re-run.

---

## 7.7 Known Failure: JSONDecodeError with Local Models

This is the most common issue when using local models (like llama3.2) as a deepeval judge. Understanding it is important for SDET interviews.

### What Happens

deepeval sends the judge model a structured prompt like:
```
"Rate the relevancy of this answer on a scale of 0–1. Return your answer as JSON: {"score": ..., "reason": "..."}"
```

A cloud model (GPT-4, Claude) reliably returns:
```json
{"score": 0.9, "reason": "The answer directly addresses the question."}
```

A local model (llama3.2) may return any of these instead:

| Bad Output | Error Thrown |
|---|---|
| ` ```json\n{"score": 0.9}\n``` ` | `Expecting value` — newline after fence breaks parser |
| `{"score": 0.9} Here is my reasoning...` | `Extra data` — text after closing `}` |
| `{"score": 0.9, "reason": "See \HTTP spec"}` | `Invalid \escape` — `\H` is not a valid JSON escape |
| `{'score': 0.9, 'reason': 'ok'}` | `Expecting property name` — single quotes are not valid JSON |

### The Two-Layer Fix in `app/local_judge.py`

**Layer 1 — Ollama JSON mode (at the model level):**
```python
response = _ollama_lib.chat(
    model=self.model_name,
    messages=[...],
    format="json",      # ← grammar-based constraint; forces valid JSON tokens
)
```
`format="json"` tells Ollama's grammar engine to constrain the model's token sampling so it can only emit tokens that form valid JSON. This prevents most bad outputs at the source.

**Layer 2 — `_extract_json()` response cleaner (as fallback):**
```python
text = self._extract_json(response.message.content)
```
Even with JSON mode, some edge cases slip through. `_extract_json()` applies these fixes in order:
1. Strips markdown code fences (` ```json ``` `)
2. Tries `json.loads()` on the cleaned text
3. Uses bracket matching to extract just the first complete `{...}` block
4. Calls `_fix_json_string()` which fixes invalid escapes (`\H` → `\\H`), trailing commas, single quotes

### The `context` vs `retrieval_context` Bug

A separate but related failure came from using the wrong field name in `LLMTestCase`.

deepeval uses two distinct fields:
- `retrieval_context` — the documents passed to the LLM (for faithfulness/hallucination checks)
- `context` — does **not** exist as a standard field in `LLMTestCase`

**Wrong (causes `validate_assert_test_inputs` failure):**
```python
test_case = LLMTestCase(
    input=result["input"],
    actual_output=result["output"],
    context=[context],             # ← wrong field name
)
```

**Correct:**
```python
test_case = LLMTestCase(
    input=result["input"],
    actual_output=result["output"],
    retrieval_context=[context],   # ← correct field name
)
```
This also needed to be fixed in `NoHallucinatedNumberMetric` which was reading `test_case.context` — changed to `test_case.retrieval_context`.

### Interview Talking Point

> "We observed JSONDecodeErrors when using llama3.2 as a local judge — the model was adding markdown fences, prose after the JSON, and invalid escape sequences. We fixed this with two layers: Ollama's `format='json'` grammar constraint which prevents bad tokens at the model level, and a post-processing cleaner that extracts and repairs the JSON as a fallback. We also caught a field-name bug — deepeval uses `retrieval_context` not `context` in LLMTestCase, which we discovered by reading the failure logs in the HTML report."

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
