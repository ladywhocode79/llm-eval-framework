### 13.2 Challenge 2: Designing Context Variants to Test Different RAG Pipeline Stages

A single static `retrieved_context` blob only tests one point in the pipeline. In a real RAG system, what ends up in context depends entirely on *where* filtering happens — at the vector retriever, at a deterministic DB/tool-call layer, or nowhere at all. Testing only one layout leaves the others unverified. We split one scenario into **three deliberate variants** in `golden_set.json`:

| Variant | What It Simulates | QA Objective |
|---------|--------------------|---------------|
| **1 — Unfiltered** (`MP_SEC_002_VAR1_UNFILTERED`) | Retriever returned top-K candidates by semantic similarity alone, including an unsafe peanut recipe | Verify the **LLM itself** reasons about and excludes the unsafe recipe |
| **2 — Pre-filtered** (`MP_SEC_002_VAR2_PREFILTERED`) | The DB/tool-call layer already applied `exclude_allergens=["peanuts"]` before the LLM ever saw the context | Verify the LLM summarizes the safe recipe **without inventing unsubstantiated safety claims** |
| **3 — Poisoned / no safe option** (`MP_SEC_002_VAR3_POISONED`) | Retriever returned *only* recipes containing the allergen — no safe match exists | Verify the LLM **refuses** to recommend anything rather than hallucinating a "safe" pick |

Each variant exercises a different failure mode: Variant 1 tests reasoning/exclusion, Variant 2 tests over-claiming, Variant 3 tests refusal behavior under a guardrail failure upstream.

> **Interview talking point:** *"Instead of writing one test case per user scenario, we modeled the same allergy scenario across three RAG pipeline states — unfiltered retrieval, pre-filtered retrieval, and a 'poisoned' context with no safe options. That let us verify three different responsibilities: does the LLM reason correctly when handed unsafe data, does it avoid over-claiming when handed safe data, and does it refuse gracefully when there's genuinely no safe answer."*

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
