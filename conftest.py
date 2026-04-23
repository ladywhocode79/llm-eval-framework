"""
Shared pytest fixtures for the LLM eval framework.
"""

import json
import os
import sys
import pytest
from dotenv import load_dotenv
from app.qa_pipeline import QAPipeline

load_dotenv()

DATASET_PATH = os.path.join(os.path.dirname(__file__), "evals", "datasets", "qa_test_cases.json")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_ollama_package() -> None:
    """Fail fast with a clear message if the ollama package is missing."""
    try:
        import ollama  # noqa: F401
    except ImportError:
        pytest.exit(
            "\n\n"
            "  [Setup Error] The 'ollama' Python package is not installed.\n"
            "  Fix:\n\n"
            "      pip install -r requirements.txt\n\n"
            "  Then re-run your tests.\n",
            returncode=1,
        )


def _check_ollama_server(model: str) -> None:
    """Fail fast with a clear message if Ollama is not running or model is missing."""
    import ollama

    # 1. Check server is reachable
    try:
        pulled_models = [m.model for m in ollama.list().models]
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower() or "connect" in str(e).lower():
            pytest.exit(
                "\n\n"
                "  [Setup Error] Cannot connect to the Ollama server.\n"
                "  Fix: open a new terminal and run:\n\n"
                "      ollama serve\n\n"
                "  Keep it running, then re-run your tests.\n"
                "  If Ollama is not installed:\n"
                "      Mac:   brew install ollama\n"
                "      Other: https://ollama.com/download\n",
                returncode=1,
            )
        pytest.exit(
            f"\n\n  [Setup Error] Unexpected Ollama error: {e}\n",
            returncode=1,
        )

    # 2. Check the required model has been pulled
    # Model names may include a tag suffix (e.g. "llama3.2:latest"), so match on prefix
    model_base = model.split(":")[0]
    model_found = any(
        m.split(":")[0] == model_base for m in pulled_models
    )
    if not model_found:
        pytest.exit(
            f"\n\n"
            f"  [Setup Error] Model '{model}' has not been pulled yet.\n"
            f"  Fix: run the following command (one-time, ~2 GB download):\n\n"
            f"      ollama pull {model}\n\n"
            f"  Then re-run your tests.\n"
            f"  Available alternatives:\n"
            f"      ollama pull llama3.2:1b   # lightest (~1 GB)\n"
            f"      ollama pull mistral        # highest quality (~4 GB)\n"
            f"\n"
            f"  Currently pulled models: {pulled_models or '(none)'}\n",
            returncode=1,
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline():
    """Shared QA pipeline instance (one Claude client for the whole test session)."""
    return QAPipeline()


@pytest.fixture(scope="session")
def judge():
    """
    LLM judge used by deepeval metrics (AnswerRelevancyMetric, FaithfulnessMetric, etc.).

    Controlled via environment variables:
      JUDGE_BACKEND=ollama   → local Ollama model (default, free)
      JUDGE_BACKEND=openai   → OpenAI via deepeval default (requires OPENAI_API_KEY)

    Ollama model is set via:
      OLLAMA_MODEL=llama3.2  (default)

    Returns None for the openai backend — deepeval will use its built-in default.
    """
    backend = os.environ.get("JUDGE_BACKEND", "ollama").lower()

    if backend == "ollama":
        _check_ollama_package()
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        _check_ollama_server(model)

        from app.local_judge import OllamaJudge
        print(f"\n  [judge] Using local Ollama judge: {model}")
        return OllamaJudge(model=model)

    print("\n  [judge] Using deepeval default judge (OpenAI)")
    return None


@pytest.fixture(scope="session")
def test_dataset():
    """Load all test cases from the JSON dataset."""
    with open(DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def dataset_by_category(test_dataset):
    """Return test cases grouped by category."""
    grouped = {}
    for tc in test_dataset:
        cat = tc.get("category", "uncategorized")
        grouped.setdefault(cat, []).append(tc)
    return grouped
