import json
import os
import time
import logging
import pytest
from sklearn.metrics import cohen_kappa_score
from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from app.judge_factory import ClaudeLLM
from .test_evals import allergen_safety_metric, load_calibration_dataset

logger = logging.getLogger(__name__)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5-5"

# Price per token (USD). Sonnet rate is an assumption - verify against current pricing.
PRICING = {
    HAIKU: {"input": 1.00 / 1e6, "output": 5.00 / 1e6},
    SONNET: {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
}

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports")


def save_benchmark_report(results: list):
    """Persist results to reports/ (stdout is not captured in report.html)."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "model_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "| Model | κ | Total (s) | p50 (ms) | p95 (ms) | Est. cost (USD) |",
        "|-------|---|-----------|----------|----------|-----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model_name']} | {r['kappa_score']:.2f} | {r['total_time_sec']:.1f} | "
            f"{r['p50_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} | {r['estimated_cost_usd']:.6f} |"
        )
    with open(os.path.join(REPORT_DIR, "model_benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_model_benchmark(model_name: str, dataset: list):
    """Executes evaluation suite over a dataset and collects latency, accuracy, and estimated cost."""
    judge_llm = ClaudeLLM(model_name=model_name)
    
    # Custom GEval metric initialized with target judge model
    safety_metric = GEval(
        name="Allergen Safety Metric",
        criteria=allergen_safety_metric.criteria,  # same rubric as the real suite
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        threshold=0.85,
        model=judge_llm
    )

    human_labels = []
    judge_labels = []
    latencies = []
    
    # Estimate token usage (average prompt ~350 input tokens, completion ~150 output tokens)
    estimated_input_tokens = 0
    estimated_output_tokens = 0

    start_total_time = time.perf_counter()

    for item in dataset:
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["actual_output"],
            retrieval_context=item["retrieved_context"]
        )

        t0 = time.perf_counter()
        result = deepeval_evaluate(
            [test_case],
            [safety_metric],
            display_config=DisplayConfig(show_indicator=False, print_results=False)
        )
        t1 = time.perf_counter()

        elapsed = (t1 - t0) * 1000  # Convert to milliseconds
        latencies.append(elapsed)

        test_result = result.test_results[0]
        judge_passed = 1 if test_result.success else 0

        human_labels.append(item["human_label"])
        judge_labels.append(judge_passed)

        # Accumulate token estimates
        estimated_input_tokens += 350
        estimated_output_tokens += 150

    total_time_sec = time.perf_counter() - start_total_time
    kappa = cohen_kappa_score(human_labels, judge_labels)
    
    # Cost calculation
    rates = PRICING[model_name]
    cost = (estimated_input_tokens * rates["input"]) + (estimated_output_tokens * rates["output"])

    latencies.sort()
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    return {
        "model_name": model_name,
        "kappa_score": kappa,
        "total_time_sec": total_time_sec,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "estimated_cost_usd": cost,
        "total_runs": len(dataset)
    }

@pytest.mark.benchmark
def test_compare_haiku_vs_sonnet():
    dataset = load_calibration_dataset()
    
    logger.info("Starting Benchmark for Claude Haiku 4.5...")
    haiku_res = run_model_benchmark(HAIKU, dataset)

    logger.info("Starting Benchmark for Claude Sonnet 5.5...")
    sonnet_res = run_model_benchmark(SONNET, dataset)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("      LLM JUDGE BENCHMARK: CLAUDE HAIKU 4.5 vs. CLAUDE SONNET 5.5")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Claude Haiku 4.5':<22} | {'Claude Sonnet 5.5':<22}")
    print("-" * 80)
    print(f"{'Accuracy (Cohen Kappa κ)':<25} | {haiku_res['kappa_score']:<22.4f} | {sonnet_res['kappa_score']:<22.4f}")
    print(f"{'Total Execution Time':<25} | {haiku_res['total_time_sec']:<20.2f}s | {sonnet_res['total_time_sec']:<20.2f}s")
    print(f"{'p50 Latency (per test)':<25} | {haiku_res['p50_latency_ms']:<20.2f}ms | {sonnet_res['p50_latency_ms']:<20.2f}ms")
    print(f"{'p95 Latency (per test)':<25} | {haiku_res['p95_latency_ms']:<20.2f}ms | {sonnet_res['p95_latency_ms']:<20.2f}ms")
    print(f"{'Estimated Cost (10 runs)':<25} | ${haiku_res['estimated_cost_usd']:<21.6f} | ${sonnet_res['estimated_cost_usd']:<21.6f}")
    print("=" * 80)

    save_benchmark_report([haiku_res, sonnet_res])

    # Verification assertions (checked after the report is saved so one model's
    # failure never hides the other model's numbers)
    failures = [
        f"{r['model_name']} failed accuracy gate: κ={r['kappa_score']:.2f}"
        for r in (haiku_res, sonnet_res)
        if r["kappa_score"] < 0.80
    ]
    assert not failures, "; ".join(failures)
