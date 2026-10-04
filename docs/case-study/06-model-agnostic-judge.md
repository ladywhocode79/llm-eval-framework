### 13.6 Challenge 6: Making the Judge Model-Agnostic — When "Local" Isn't Applicable

Every earlier challenge in this section was fixed using Claude as the judge. That was hardcoded: `test_evals.py` imported `Anthropic`/`AsyncAnthropic` directly and built a `ClaudeLLM` wrapper with no way to swap providers. The fix was a small `get_judge_model()` factory (`app/judge_factory.py`) that resolves a judge at runtime instead of hardcoding one:

```python
backend = os.getenv("EVAL_JUDGE_BACKEND", os.getenv("JUDGE_BACKEND", "auto")).lower()

if backend == "auto":
    if _ollama_available(ollama_model):       # local, free, reachable + model pulled?
        backend = "ollama"
    elif os.getenv("GEMINI_API_KEY"):
        backend = "gemini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        backend = "anthropic"
```

The stated goal was: prefer a local model (no API cost, no data leaving the machine) wherever it's actually good enough, and fall back to a cloud provider — Gemini or Claude, whichever key is present in `.env` — otherwise. "Auto" tries local first, but an explicit `EVAL_JUDGE_BACKEND` always wins, so a specific backend can still be forced.

**Testing this immediately produced real evidence, not just theory.** Running the exact same four scenarios and the exact same `allergen_safety_metric` rubric through the local `llama3.2` judge instead of Claude:

| Scenario | Claude judge | llama3.2 judge |
|----------|--------------|-----------------|
| `MP_VAL_001` | Faithfulness 1.0, PASS | Faithfulness **0.33** — "no contradictions... indicating a lack of faithful match" (self-contradictory reasoning) |
| `MP_SEC_002_VAR2_PREFILTERED` | Allergen Safety 0.9, PASS | Allergen Safety 0.8, but reasoning claims *"Grilled Salmon... actually contains peanuts"* — factually false; the context never says that |
| `MP_SEC_002_VAR2_PREFILTERED` | Relevancy 1.0, PASS | Relevancy **0.0** — *"output provided general dinner suggestions without considering the specific allergy"*, despite the output correctly excluding peanuts |

The local model wasn't just scoring more harshly — it was **hallucinating facts about the retrieval context** while grading. That's a materially different failure mode than the earlier judicial-drift/temperature issues: no amount of prompt scoping fixes a judge that misreads the input it's grading.

**The fix wasn't more prompt engineering — it was routing this specific test file to a stronger judge**, while leaving the *other* eval files (`test_answer_relevancy.py`, `test_faithfulness.py`, which ask simpler, single-fact questions) on the local Ollama judge via the existing `JUDGE_BACKEND` fixture in `conftest.py`. To avoid the two systems colliding — `conftest.py` accepts `ollama`/`openai`, this factory accepts `ollama`/`gemini`/`anthropic`/`auto` — the factory reads a separate `EVAL_JUDGE_BACKEND` first and only falls back to the shared `JUDGE_BACKEND` if that's unset:

```bash
# .env
JUDGE_BACKEND=ollama          # test_answer_relevancy.py / test_faithfulness.py: local is fine
EVAL_JUDGE_BACKEND=anthropic  # test_evals.py: safety rubric needs stronger reasoning
```

> **Interview talking point:** *"We made the judge model-agnostic rather than hardcoding Claude, with a stated preference for a free local judge wherever it's good enough. Testing that preference immediately produced evidence instead of assumption: on our safety-critical allergen rubric, the local llama3.2 judge didn't just score more conservatively than Claude — it hallucinated facts about the retrieval context while grading it, like claiming a recipe contained peanuts when it didn't. That's not a prompt-engineering problem, it's a capability ceiling. So the right call was routing that one test file to a stronger cloud judge while keeping the simpler relevancy/faithfulness checks on the free local judge — and building the config so both choices coexist without one silently overriding the other."*

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
