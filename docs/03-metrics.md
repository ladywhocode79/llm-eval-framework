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

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
