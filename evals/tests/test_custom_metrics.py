"""
Eval: Custom Metrics
Tests using deterministic (non-LLM-judge) custom metrics:
  - KeywordPresentMetric
  - OutputLengthMetric
  - NoHallucinatedNumberMetric
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from evals.metrics.custom_metrics import (
    KeywordPresentMetric,
    OutputLengthMetric,
    NoHallucinatedNumberMetric,
)


@pytest.mark.eval
class TestCustomMetrics:

    # ── Keyword presence ──────────────────────────────────────────────────────

    def test_keyword_capital_paris(self, pipeline):
        result = pipeline.answer(
            question="What is the capital of France?",
            context="France is a country in Western Europe. Its capital is Paris.",
        )
        metric = KeywordPresentMetric(keywords=["paris"], threshold=1.0)
        test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
        assert_test(test_case, [metric])

    def test_keyword_http_protocol(self, pipeline):
        result = pipeline.answer(
            question="What does HTTP stand for?",
            context="HTTP stands for HyperText Transfer Protocol.",
        )
        metric = KeywordPresentMetric(keywords=["hypertext", "transfer", "protocol"], threshold=1.0)
        test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
        assert_test(test_case, [metric])

    # ── Output length ─────────────────────────────────────────────────────────

    def test_output_not_empty(self, pipeline):
        result = pipeline.answer(question="What is 2 + 2?")
        metric = OutputLengthMetric(min_words=1, max_words=200)
        test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
        assert_test(test_case, [metric])

    def test_summary_concise(self, pipeline):
        result = pipeline.answer(
            question="In one sentence, what is Python?",
            context="Python is a high-level, interpreted programming language known for its readability.",
        )
        metric = OutputLengthMetric(min_words=5, max_words=60)
        test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
        assert_test(test_case, [metric])

    # ── No hallucinated numbers ───────────────────────────────────────────────

    def test_boiling_point_no_hallucinated_numbers(self, pipeline):
        context = "Water boils at 100 degrees Celsius at 1 atm."
        result = pipeline.answer(question="What is the boiling point of water?", context=context)
        metric = NoHallucinatedNumberMetric(threshold=1.0)
        test_case = LLMTestCase(
            input=result["input"],
            actual_output=result["output"],
            retrieval_context=[context],
        )
        assert_test(test_case, [metric])

    def test_apple_math_no_hallucinated_numbers(self, pipeline):
        result = pipeline.answer(
            question="A store has 50 apples and sells 18. How many remain?",
            context="The store started with 50 apples and sold 18 of them.",
        )
        metric = NoHallucinatedNumberMetric(threshold=1.0)
        test_case = LLMTestCase(
            input=result["input"],
            actual_output=result["output"],
            retrieval_context=[result["context"]],
        )
        assert_test(test_case, [metric])

    # ── Combined metrics ──────────────────────────────────────────────────────

    def test_combined_keyword_and_length(self, pipeline):
        result = pipeline.answer(
            question="What does REST stand for?",
            context="REST stands for Representational State Transfer.",
        )
        test_case = LLMTestCase(input=result["input"], actual_output=result["output"])
        assert_test(test_case, [
            KeywordPresentMetric(keywords=["representational", "state", "transfer"]),
            OutputLengthMetric(min_words=3, max_words=100),
        ])
