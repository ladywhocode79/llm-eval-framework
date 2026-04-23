"""
Eval: Faithfulness (Hallucination Detection)
Tests that the model's output is faithful to the provided context.
Uses deepeval's FaithfulnessMetric with a local Ollama judge (default) or OpenAI.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.eval
class TestFaithfulness:
    """Faithfulness evals — does the output stay grounded in the context?"""

    THRESHOLD = 0.7

    def _run(self, pipeline, judge, question: str, context: str):
        result = pipeline.answer(question=question, context=context)
        metric = FaithfulnessMetric(
            threshold=self.THRESHOLD,
            model=judge,          # None = deepeval default (OpenAI)
            verbose_mode=True,
        )
        test_case = LLMTestCase(
            input=result["input"],
            actual_output=result["output"],
            retrieval_context=[context],
        )
        assert_test(test_case, [metric])

    def test_boiling_point_grounded(self, pipeline, judge):
        """Model should stick to the context and not hallucinate temperature values."""
        self._run(
            pipeline, judge,
            question="What is the boiling point of water?",
            context="Water boils at 100 degrees Celsius at standard atmospheric pressure (1 atm).",
        )

    def test_world_cup_out_of_context(self, pipeline, judge):
        """Model should admit it doesn't know rather than hallucinate."""
        self._run(
            pipeline, judge,
            question="Who won the 2050 World Cup?",
            context="The 2022 FIFA World Cup was held in Qatar. Argentina won by defeating France in the final.",
        )

    def test_rest_api_faithful(self, pipeline, judge):
        self._run(
            pipeline, judge,
            question="What HTTP methods do REST APIs use?",
            context=(
                "REST APIs use standard methods like GET, POST, PUT, and DELETE. "
                "They are stateless, meaning each request is self-contained."
            ),
        )

    @pytest.mark.parametrize("tc_id", ["tc_001", "tc_002", "tc_004", "tc_005", "tc_006"])
    def test_dataset_faithfulness(self, pipeline, judge, test_dataset, tc_id):
        """Drive faithfulness tests from the JSON dataset (context-based cases only)."""
        tc = next(t for t in test_dataset if t["id"] == tc_id)
        if not tc.get("context"):
            pytest.skip(f"{tc_id} has no context — faithfulness test not applicable.")
        self._run(pipeline, judge, question=tc["question"], context=tc["context"])
