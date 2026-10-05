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

---

[← Back to the guide index](../LLM_TESTING_GUIDE.md)
