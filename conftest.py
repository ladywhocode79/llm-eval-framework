"""
Shared pytest fixtures for the LLM eval framework.
"""

import json
import os
import pytest
from dotenv import load_dotenv
from app.qa_pipeline import QAPipeline

load_dotenv()

DATASET_PATH = os.path.join(os.path.dirname(__file__), "evals", "datasets", "qa_test_cases.json")


@pytest.fixture(scope="session")
def pipeline():
    """Shared QA pipeline instance (one Claude client for the whole test session)."""
    return QAPipeline()


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
