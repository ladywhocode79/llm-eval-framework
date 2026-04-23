"""
Custom deepeval metrics for the LLM eval framework.
"""

import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class KeywordPresentMetric(BaseMetric):
    """Checks that all required keywords appear in the output (case-insensitive)."""

    def __init__(self, keywords: list[str], threshold: float = 1.0):
        self.keywords = [k.lower() for k in keywords]
        self.threshold = threshold
        self.score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output.lower()
        found = [kw for kw in self.keywords if kw in output]
        self.score = len(found) / len(self.keywords) if self.keywords else 1.0
        self.success = self.score >= self.threshold

        missing = [kw for kw in self.keywords if kw not in output]
        self.reason = (
            f"All keywords found." if not missing
            else f"Missing keywords: {missing}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def name(self) -> str:
        return "KeywordPresentMetric"


class OutputLengthMetric(BaseMetric):
    """Checks that the output length (words) is within an acceptable range."""

    def __init__(self, min_words: int = 1, max_words: int = 300, threshold: float = 1.0):
        self.min_words = min_words
        self.max_words = max_words
        self.threshold = threshold
        self.score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        word_count = len(test_case.actual_output.split())
        in_range = self.min_words <= word_count <= self.max_words
        self.score = 1.0 if in_range else 0.0
        self.success = in_range
        self.reason = (
            f"Word count {word_count} is within [{self.min_words}, {self.max_words}]."
            if in_range
            else f"Word count {word_count} is outside [{self.min_words}, {self.max_words}]."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def name(self) -> str:
        return "OutputLengthMetric"


class NoHallucinatedNumberMetric(BaseMetric):
    """
    Checks that any numbers in the output also appear in the context.
    Useful for catching hallucinated figures.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", test_case.actual_output))
        context = test_case.retrieval_context or []
        context_text = " ".join(context) if isinstance(context, list) else (context or "")
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context_text))

        # Allow numbers not in context only if context is empty (no-context questions)
        if not context_text.strip():
            self.score = 1.0
            self.success = True
            self.reason = "No context provided; number check skipped."
            return self.score

        hallucinated = output_numbers - context_numbers
        self.score = 1.0 if not hallucinated else 0.0
        self.success = not hallucinated
        self.reason = (
            "All numbers in output are grounded in context."
            if not hallucinated
            else f"Potentially hallucinated numbers: {hallucinated}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def name(self) -> str:
        return "NoHallucinatedNumberMetric"
