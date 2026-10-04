## 14. Glossary

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
| **Ollama** | A tool to run open-source LLMs locally; exposes them via a REST API on localhost |
| **DeepEvalBaseLLM** | deepeval's abstract base class for plugging in any LLM as a judge |
| **JUDGE_BACKEND** | Env var that controls whether the judge uses Ollama (local) or OpenAI (cloud) |
| **SDET** | Software Development Engineer in Test — engineers who build test frameworks and automation |
| **GEval** | deepeval's framework for building custom, natural-language-criteria LLM-as-judge metrics |
| **Judicial Drift / Over-generalization** | When an LLM judge ignores your specific evaluation criteria and falls back on its own broad commonsense notion of "quality" or "safety" |
| **Hallucination of Omission** | An unverifiable, additive claim an LLM makes that isn't contradicted by context but also isn't supported by it — missed by contradiction-based metrics like Faithfulness |
| **Judge Calibration** | Measuring how closely an LLM judge's verdicts match human expert labels on a gold set, before trusting it as a test oracle |
| **Cohen's Kappa (κ)** | Agreement statistic between two raters that corrects for chance agreement; ≥ 0.80 is treated as strong agreement here |
| **Gold Set (Human-Annotated Set)** | A small dataset with human-assigned labels and reasoning, used as ground truth for calibrating a judge |
| **Tool-Call Schema Validation** | Deterministically validating an agent's function/tool-call arguments (e.g. with Pydantic) instead of trusting free-text output alone |

---

*Framework built with: Python · deepeval · Anthropic Claude SDK · pytest*

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
