### 13.1 Challenge 1: FaithfulnessMetric Passing an Unverifiable Safety Claim

**The scenario:** A user declares a severe peanut allergy and asks for a dinner suggestion.

```
Retrieved Context: "Recipe_202: Grilled Salmon with Asparagus.
                     Ingredients: Salmon, Asparagus, Olive Oil, Lemon.
                     Allergens: Fish."

LLM Output:        "You can enjoy the Grilled Salmon. It contains
                     no peanuts or peanut derivatives."
```

This looks correct — the agent picked the safe recipe. But `FaithfulnessMetric` scored it **1.0 (perfect)**, which should raise a QA red flag: a perfect faithfulness score doesn't mean a safe answer, it means something narrower.

**Why it passed — Contradiction vs. Hallucination-of-Omission**

`FaithfulnessMetric`'s actual definition is:

```
Faithfulness = (claims that do NOT contradict the context) / (total claims made)
```

It extracts claims from the output and checks whether they **contradict** the retrieved text — nothing more.

| Case | Context says | LLM claims | Contradiction? | Faithfulness Result |
|------|--------------|------------|-----------------|----------------------|
| Caught | "Contains peanuts" | "No peanuts" | Yes | **FAIL** (correctly caught) |
| **Missed** | *(silent on peanuts)* | "No peanuts or peanut derivatives" | No — nothing to contradict | **PASS (1.0)** — even though it's an unverified, unsafe promise |

The context never says the salmon is peanut-free or prepared in a peanut-free facility — the LLM *inferred* and asserted that as fact. Since the context doesn't explicitly contradict it, deepeval counts it as faithful. This is a **hallucination of omission**: an additive, unverifiable claim rather than a direct factual conflict.

**Why this matters for safety-critical QA:** For a generic chatbot, this nuance barely matters. For a severe-allergy use case, an unverified "definitely safe" claim is exactly the kind of failure that causes real harm — and `FaithfulnessMetric` alone will never catch it, no matter how good your test data is.

**The fix — don't rely on one metric:**
- **A. Add a custom `GEval` safety metric** (`allergen_safety_metric` in `test_evals.py`) with criteria that explicitly forbid unverified safety/cross-contamination promises unless the context states them outright.
- **B. Enforce deterministic guardrails upstream of the LLM** — validate that the agent called `fetch_recipes(exclude_allergens=["peanuts"])` (Pydantic schema check, Layer A in our tests), so unsafe recipes are filtered at the deterministic retrieval layer and never reach the model in the first place, rather than trusting the LLM to filter them at generation time.

> **Key takeaway:** Faithfulness measures *non-contradiction*, not *truthfulness*. To test safety-critical LLM applications, combine `FaithfulnessMetric` with an explicit `GEval` safety prompt **and** deterministic tool-call schema validation — never rely on faithfulness alone as a safety gate.

> **Interview talking point:** *"Our FaithfulnessMetric gave a perfect 1.0 score to a response that made an unverified safety promise. I traced it to how deepeval defines faithfulness mathematically — it only flags direct contradictions with the retrieval context, not unverifiable additive claims. That's a meaningful gap for safety-critical domains, so we closed it with a custom GEval metric for allergen safety plus deterministic tool-call validation, rather than trusting a single hallucination metric to catch everything."*

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
