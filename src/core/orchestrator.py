import logging
from pathlib import Path
import re

from src.core.decorators import log_execution
from src.data.csv_processor import CsvDataProcessor
from src.llms.llm_classifier import LLMClassifier
from src.llms.llm_translator import LLMTranslator
from src.parsing.query_parser import QueryParser
from src.utils.normalization import normalize_classification


logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
DATA_PATH = project_root / "data_source" / "freelancer_earnings_bd.csv"


class FreelancerQueryOrchestrator:
    """
    Orchestrates the process of analyzing a freelancer data query, involving
    translation, classification, parsing, and data processing.
    """

    # Default target language for analysis (after translation)
    ANALYSIS_LANGUAGE = "en"

    def __init__(self, data_processor, classifier, translator, parser):
        """
        Initializes the orchestrator with its necessary parts.

        Args:
            data_processor (CsvDataProcessor): Instance for data operations.
            classifier (LLMClassifier): Instance for query classification.
            translator (LLMTranslator): Instance for query translation.
            parser (QueryParser): Instance for parsing classified queries.
        """
        if not all(
            [data_processor, classifier, parser]
        ):  # Translator can be None if not used
            raise ValueError(
                "CsvDataProcessor, LLMClassifier, and QueryParser are required."
            )

        self.data_processor = data_processor
        self.classifier = classifier
        self.translator = (
            translator  # Can be None if no translation is needed/supported
        )
        self.parser = parser
        logger.info("FreelancerQueryOrchestrator initialized with all components.")

    @log_execution()
    def process_query(
        self, raw_query_text: str, source_language_code: str = None
    ) -> str:
        """
        Processes a raw user query from start to finish.

        1. Detects language (simple heuristic) or uses provided source_language_code.
        2. Translates to English if necessary and if translator is available.
        3. Classifies the (translated) query.
        4. Normalizes the classification.
        5. Parses parameters based on classification.
        6. Calls the appropriate data processing method.
        7. Returns the result.

        Args:
            raw_query_text (str): The user's query in its original language.
            source_language_code (str, optional):
                The language code of the raw_query_text (e.g., "ru").
                If None, simple detection (Cyrillic for "ru") is attempted.

        Returns:
            str: The analytical result or an error/informational message.
        """
        if not raw_query_text or not raw_query_text.strip():
            return "Please provide a query."

        logger.info(
            f"Processing raw query: '{raw_query_text}'"
            f" (source_lang: {source_language_code or 'auto-detect'})"
        )

        # 1. Translation
        query_for_analysis = raw_query_text
        actual_source_lang = source_language_code

        if not actual_source_lang:  # Attempt simple detection if not provided
            if re.search(
                "[\u0400-\u04ff]", raw_query_text
            ):  # Basic Cyrillic check for Russian
                actual_source_lang = "ru"
            else:
                actual_source_lang = (
                    self.ANALYSIS_LANGUAGE
                )  # Assume it's already in analysis language

        if self.translator and actual_source_lang != self.ANALYSIS_LANGUAGE:
            logger.info(
                f"Translating query from '{actual_source_lang}'"
                f" to '{self.ANALYSIS_LANGUAGE}'."
            )
            translated_query = self.translator.translate(
                raw_query_text,
                source_lang=actual_source_lang,
                target_lang=self.ANALYSIS_LANGUAGE,
            )
            if (
                translated_query and translated_query != raw_query_text
            ):  # Check if the translation actually happened and was successful
                query_for_analysis = translated_query
                logger.info(f"Translated query for analysis: '{query_for_analysis}'")
            elif not translated_query:
                logger.warning(
                    f"Translation from '{actual_source_lang}' failed."
                    f" Proceeding with original query."
                )
                # query_for_analysis remains raw_query_text
            else:  # translated_query == raw_query_text (e.g., original on error)
                logger.info(
                    f"Translator returned original query. Proceeding with:"
                    f" '{query_for_analysis}'"
                )
        else:
            logger.info(
                f"No translation needed or translator not available. "
                f"Analyzing query as is: '{query_for_analysis}'"
            )

        # 2. Classification
        if not self.classifier:  # Should have been caught in __init__
            logger.error("Classifier not available. Cannot process query.")
            return "Error: System component (Classifier) missing."

        raw_classification = self.classifier.classify(query_for_analysis)
        if raw_classification is None:
            logger.error(f"Query classification failed for: '{query_for_analysis}'")
            return (
                "I'm having trouble understanding the type of your query."
                " Please try rephrasing."
            )

        logger.info(
            f"Raw LLM classification: '{raw_classification}' for query:"
            f" '{query_for_analysis}'"
        )

        # 3. Normalization
        classification = normalize_classification(raw_classification)
        logger.info(f"Normalized classification: '{classification}'")

        # 4. Parsing and Data Processing
        result_message = (
            f"I couldn't understand that query type ('{classification}')."
            f" Please try rephrasing."
        )
        params = None

        if classification == "income_comparison":
            params = self.parser.parse_income_comparison(query_for_analysis)
            if params:
                result_message = self.data_processor.compare_income(
                    params["group1_desc"], params["group2_desc"]
                )
            else:
                result_message = (
                    "Could not parse the groups for income comparison."
                    " Please specify two distinct groups "
                    "(e.g., 'X versus Y')."
                )

        elif classification == "income_distribution":
            params = self.parser.parse_income_distribution(query_for_analysis)
            if params:
                result_message = self.data_processor.get_income_distribution(
                    params["by_column"]
                )
            else:  # Should not happen if parser defaults
                result_message = (
                    "Could not determine how to distribute income. "
                    "Try 'distribution by region'."
                )

        elif classification == "percentage_calculation":
            params = self.parser.parse_percentage_calculation(query_for_analysis)
            if params:
                result_message = self.data_processor.calculate_percentage(
                    params["condition"], params["criteria"]
                )
            else:
                result_message = (
                    "Could not parse the details for percentage"
                    " calculation. Please specify 'percentage of [group]"
                    " who [did something]'."
                )

        elif classification == "average_income_calculation":
            params = self.parser.parse_average_income(query_for_analysis)
            if params:
                result_message = self.data_processor.get_single_group_average_income(
                    params["group_description"]
                )
            else:
                result_message = (
                    "Could not parse the group for average income"
                    " calculation. Please specify 'average income"
                    " of [group]'."
                )

        elif classification == "mode_calculation":
            params = self.parser.parse_mode_calculation(query_for_analysis)
            if params and "column_name" in params:
                # The parser should return the column name that CsvDataProcessor expects
                # This might be the original name ("Job_Category") or
                # a cleaned name ("job_category")
                # depending on your CsvDataProcessor.get_most_common implementation.
                result_message = self.data_processor.get_most_common(
                    params["column_name"]
                )
            else:
                result_message = (
                    "Could not determine which item's frequency"
                    " to calculate. Please specify (e.g.,"
                    " 'most common job category')."
                )

        elif classification == "list_unique_values":
            params = self.parser.parse_list_unique_values(query_for_analysis)
            if params and "column_name" in params:
                result_message = self.data_processor.get_unique_values(
                    params["column_name"]
                )
            else:
                result_message = (
                    "Could not determine which category's unique values"
                    " to list. Please specify "
                    "(e.g., 'list all job categories')."
                )

        elif classification == "other":
            result_message = (
                "I'm not sure how to answer that specific query"
                " with the available data tools. Please try asking about"
                " income comparisons, distributions, percentages,"
                " or average incomes."
            )

        # logger.info(f"Final result for query '{raw_query_text}': {result_message}")
        return result_message


# --- Example Main/CLI integration ---
if __name__ == "__main__":
    # 1. Initialize components
    try:
        data_proc = CsvDataProcessor(data_path=str(DATA_PATH))
        # Specify a device if needed, e.g., for MPS on Mac
        import torch

        device = (
            "mps"
            if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
            else "cpu"
        )
        #  Or pass None if no translation desired
        translator = LLMTranslator(device=device)
        classifier = LLMClassifier(device=device)
        parser = QueryParser()

        # 2. Create Orchestrator
        orchestrator = FreelancerQueryOrchestrator(
            data_processor=data_proc,
            classifier=classifier,
            translator=translator,
            parser=parser,
        )

        # 3. Process queries (example)
        queries_to_test = [
            ("What is the average income of experts?", "en"),
            ("Compare income of PayPal users vs crypto users", "en"),
            ("Какой процент экспертов выполнил менее 10 проектов?", "ru"),
            ("Show income distribution by payment method", "en"),
            ("This is an unknown query type", "en"),
        ]

        for q_text, q_lang in queries_to_test:
            logger.info(f"\n>>> Query ({q_lang}): {q_text}")
            response = orchestrator.process_query(q_text, source_language_code=q_lang)
            logger.info(f"<<< Response: {response}")

    except Exception as e:
        print(
            f"An error occurred during orchestrator setup or query processing."
            f"A critical error occurred: {e}"
        )
