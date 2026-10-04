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

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
