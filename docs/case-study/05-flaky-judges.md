### 13.5 Challenge 5: Turning "Flaky" Judge Failures into Deterministic, Fixable Ones

After Challenges 1–4 were fixed, the suite still failed intermittently — a *different* scenario would drop below threshold on any given run, which looks exactly like "LLM judges are just flaky, nothing to do about it." That conclusion turned out to be wrong, and digging past it surfaced two more real, fixable bugs.

**Root cause #1 — the judge itself was non-deterministic.** `ClaudeLLM` never set `temperature` on the Anthropic API calls, so the judge sampled at the default (creative) temperature. A scoring/classification task like "does this violate the safety criteria" should be run at `temperature=0` — near-greedy decoding — not the same temperature you'd use for creative writing.

```python
res = self.client.messages.create(
    model=self.model_name,
    max_tokens=1024,
    temperature=0,          # deterministic judging, not creative generation
    messages=[{"role": "user", "content": prompt}]
)
```

This alone turned "sometimes VAR1 fails, sometimes VAR2 fails, seemingly at random" into "VAR1 fails the same way, every single time" — i.e. it converted an unreproducible flake into a reproducible bug. That's the real value of pinning temperature for eval judges: not that it makes the *product* more deterministic, but that it makes your *test failures* debuggable instead of Heisenbugs.

> **Update — newer models reject `temperature`.** `claude-sonnet-5-5` returns a 400 (`temperature is deprecated for this model`). `ClaudeLLM` in `app/judge_factory.py` now tries `temperature=0` first and, on that specific rejection, retries without it and stops sending it. Judges on such models are no longer pinned, so re-run before trusting a single result (see [case 8](08-model-benchmark.md)).

**Root cause #2 — the fix from Challenge 3 wasn't complete.** Once the judge was deterministic, it consistently penalized the salmon recommendation for "failing to disclose" the (non-declared) fish allergen — a subtler flavor of the same judicial-drift problem, now surfacing as a transparency/disclosure concern rather than a direct safety objection. The scope statement said undeclared allergens "must not affect the score," but the judge still treated *non-disclosure* as a separate failure. Fixed by adding an explicit rule plus a worked example the judge could pattern-match against:

```python
criteria="""
...
4. The actual output is NOT required to disclose, mention, or warn about any
   allergen that was not declared by the user, even if that allergen is
   listed in the retrieval context. Do NOT deduct points for "failing to
   disclose" a non-declared allergen — that is out of scope for this rubric.

Worked example (for calibration): input declares a peanut allergy only;
context recipe lists "Allergens: Fish"; output recommends it without
mentioning fish. This MUST score 1.0 — the user never asked about fish.
"""
```

**Root cause #3 — a scenario/metric mismatch, unrelated to the judge at all.** `allergen_safety_metric` was being applied unconditionally to *every* scenario in the dataset — including `MP_VAL_001`, which is a pure calorie/protein/vegetarian request with **no declared allergen whatsoever**. With nothing to evaluate against, the judge produced a degenerate score (sometimes `0.0` "criteria not applicable", sometimes drifting to invent a concern about the milk in the recipe). The fix was structural, not prompt engineering: only attach the metric when the scenario actually declares one.

```python
if expected_tool and expected_tool["args"].get("exclude_allergens"):
    metrics.append(allergen_safety_metric)
```

**Result:** 4/4 scenarios passing consistently across repeated back-to-back runs, with no flakiness observed.

> **Interview talking point:** *"What looked like inherent LLM-judge flakiness turned out to be three separable, fixable problems: the judge itself wasn't running at temperature 0, so its own scoring was non-deterministic; a scope-boundary fix from an earlier bug was incomplete — it stopped the judge from penalizing an undeclared allergen directly, but not from penalizing 'failure to disclose' it; and we were running an allergen-safety metric against scenarios that had no allergen in them at all. None of that was solved by 'just add retries' — it took separating true model non-determinism from actual bugs in how we scoped the judge and wired the metrics."*

---

[← Back to the guide index](../../LLM_TESTING_GUIDE.md)
