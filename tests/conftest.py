from pathlib import Path

import pytest

from src.core.orchestrator import FreelancerQueryOrchestrator
from src.data.csv_processor import CsvDataProcessor
from src.llms.llm_classifier import LLMClassifier
from src.llms.llm_translator import LLMTranslator
from src.parsing.query_parser import QueryParser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data_source" / "freelancer_earnings_bd.csv"


@pytest.fixture(scope="session")
def orchestrator():
    """
    Pytest fixture to initialize and provide the FreelancerQueryOrchestrator.
    """
    if not DATA_PATH.is_file():
        pytest.fail(
            f"Test data file not found at {DATA_PATH}. Ensure it exists for testing."
        )

    test_device = None

    data_processor = CsvDataProcessor(data_path=str(DATA_PATH))
    translator = LLMTranslator(device=test_device)
    classifier = LLMClassifier(device=test_device)
    query_parser = QueryParser()

    app_orchestrator = FreelancerQueryOrchestrator(
        data_processor=data_processor,
        classifier=classifier,
        translator=translator,
        parser=query_parser,
    )
    return app_orchestrator
