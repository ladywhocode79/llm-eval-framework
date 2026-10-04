import pytest
from sklearn.metrics import cohen_kappa_score
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from .test_evals import allergen_safety_metric, judge_model, load_calibration_dataset


def metrics_for(item):
    """Same metric routing as test_meal_planner_scenario, so we calibrate the real gate."""
    metrics = [
        FaithfulnessMetric(threshold=0.85, model=judge_model),
        allergen_safety_metric,
    ]
    # A valid refusal is not "relevant" to AnswerRelevancyMetric (see guide 13.4).
    if not item.get("expects_refusal"):
        metrics.append(AnswerRelevancyMetric(threshold=0.80, model=judge_model))
    return metrics


@pytest.mark.calibration
def test_judge_cohen_kappa_alignment():
    """Validates that LLM Judge aligns with Human Expert labels (κ >= 0.80)."""
    dataset = load_calibration_dataset()
    human_labels, judge_labels, disagreements = [], [], []

    for item in dataset:
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["actual_output"],
            retrieval_context=item["retrieved_context"]
        )

        result = deepeval_evaluate(
            [test_case],
            metrics_for(item),
            display_config=DisplayConfig(show_indicator=False, print_results=False)
        )

        # Judge says "pass" only if ALL routed metrics pass
        judge_passed = 1 if result.test_results[0].success else 0
        human_labels.append(item["human_label"])
        judge_labels.append(judge_passed)
        if judge_passed != item["human_label"]:
            disagreements.append(item["scenario_id"])

    kappa = cohen_kappa_score(human_labels, judge_labels)
    assert kappa >= 0.80, (
        f"Judge Calibration Failed! Cohen's Kappa ({kappa:.2f}) < 0.80. "
        f"Disagreements: {disagreements}"
    )
