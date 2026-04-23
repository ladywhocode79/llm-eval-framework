"""
Simple Q&A pipeline — the LLM app being evaluated.

Given a question and optional context (retrieved documents), it produces an answer.
This simulates a basic RAG (Retrieval-Augmented Generation) pattern.
"""

from app.llm_client import LLMClient

SYSTEM_PROMPT = """You are a helpful assistant that answers questions accurately and concisely.
When context is provided, base your answer strictly on that context.
If the context does not contain enough information to answer, say so clearly.
Do not make up facts."""


class QAPipeline:
    def __init__(self, model: str | None = None):
        self.client = LLMClient(model=model) if model else LLMClient()

    def answer(self, question: str, context: str = "") -> dict:
        """
        Returns:
            {
                "input":    the original question,
                "context":  the context passed in (if any),
                "output":   the generated answer,
            }
        """
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = f"Question: {question}"

        output = self.client.complete(prompt=prompt, system=SYSTEM_PROMPT)
        return {"input": question, "context": context, "output": output}
