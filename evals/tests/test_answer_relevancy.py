"""
Eval: Answer Relevancy
Tests that the model's output is relevant to the input question.
Uses deepeval's AnswerRelevancyMetric with a local Ollama judge (default) or OpenAI.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.eval
class TestAnswerRelevancy:
    """Answer relevancy evals — does the output actually address the question?"""

    THRESHOLD = 0.7

    def _run(self, pipeline, judge, question: str, context: str = ""):
        result = pipeline.answer(question=question, context=context)
        metric = AnswerRelevancyMetric(
            threshold=self.THRESHOLD,
            model=judge,          # None = deepeval default (OpenAI)
            verbose_mode=True,
        )
        test_case = LLMTestCase(
            input=result["input"],
            actual_output=result["output"],
        )
        assert_test(test_case, [metric])

    def test_capital_city_question(self, pipeline, judge):
        self._run(pipeline, judge,
                  question="What is the capital of France?",
                  context="France is a country in Western Europe. Its capital city is Paris.")

    def test_http_acronym(self, pipeline, judge):
        self._run(pipeline, judge,
                  question="What does HTTP stand for?",
                  context="HTTP stands for HyperText Transfer Protocol.")

    def test_math_question_no_context(self, pipeline, judge):
        self._run(pipeline, judge,
                  question="If a store has 50 apples and sells 18, how many are left?")

    def test_rest_api_summary(self, pipeline, judge):
        self._run(
            pipeline, judge,
            question="Summarize the main purpose of REST APIs.",
            context=(
                "REST (Representational State Transfer) APIs allow different software systems "
                "to communicate over HTTP. They use standard methods like GET, POST, PUT, and DELETE. "
                "REST APIs are stateless, meaning each request contains all the information needed."
            ),
        )

    @pytest.mark.parametrize("tc_id", ["tc_001", "tc_002", "tc_004"])
    def test_dataset_factual_cases(self, pipeline, judge, test_dataset, tc_id):
        """Drive tests from the JSON dataset for factual cases."""
        tc = next(t for t in test_dataset if t["id"] == tc_id)
        self._run(pipeline, judge, question=tc["question"], context=tc.get("context", ""))
