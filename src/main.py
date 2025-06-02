import argparse
import logging
from pathlib import Path

from src.core.logger_setup import configure_logging
from src.core.orchestrator import FreelancerQueryOrchestrator
from src.data.csv_processor import CsvDataProcessor
from src.llms.llm_classifier import LLMClassifier
from src.llms.llm_translator import LLMTranslator
from src.parsing.query_parser import QueryParser
from tests.queries_list import test_queries_with_lang


project_root = Path(__file__).resolve().parent.parent
DATA_PATH = project_root / "data_source" / "freelancer_earnings_bd.csv"

configure_logging()

logger = logging.getLogger(__name__)


def run_interactive_mode(orchestrator: FreelancerQueryOrchestrator):
    """Runs the CLI in interactive mode."""
    logger.info("Freelancer Analytics CLI (type 'exit' or 'quit' to quit)\n")
    logger.info("You can ask questions in English or Russian.")
    while True:
        try:
            raw_query = input("Query: ")
            if not raw_query.strip():  # Skip empty input
                continue
            if raw_query.lower() in ("exit", "quit"):
                logger.info("Exiting interactive mode.")
                break

            result = orchestrator.process_query(raw_query)
            logger.info(f"\n{result}\n")

        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode due to KeyboardInterrupt.")
            break
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred in interactive mode."
                f"An error occurred: {e}. Please try again or restart."
            )


def run_test_queries(orchestrator: FreelancerQueryOrchestrator):
    """Runs a predefined set of test queries."""
    logger.info("Starting test queries...")

    for query_text, lang_code in test_queries_with_lang:
        logger.info(f"\n>>> Query ({lang_code}): {query_text}")
        logger.info(f"Running test query ({lang_code}): {query_text}")
        result = orchestrator.process_query(query_text, source_language_code=lang_code)
        logger.info(f"<<< Response: {result}\n")
    logger.info("Test queries finished.")


def main():
    """Main function to set up components and handle CLI arguments."""
    logger.info("Application starting...")

    parser = argparse.ArgumentParser(
        description="Freelancer Analytics - Analyze freelancer earnings data"
        " using LLMs and Pandas."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query to analyze."
        " If not provided, runs in interactive mode.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,  # Let the orchestrator auto-detect or assume English
        help="Language code of the input query (e.g., 'en', 'ru')."
        " Default: auto-detect/English.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run a predefined set of test queries."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,  # Let components auto-detect (cpu, mps, cuda)
        help="Specify device for LLM models (e.g., 'cpu', 'mps', 'cuda:0')."
        " Default: auto-detect.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_PATH),  # Use the calculated DATA_PATH as default
        help=f"Path to the freelancer earnings CSV data file. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Update root logger level based on CLI arg
    logging.getLogger().setLevel(args.log_level.upper())
    logger.info(f"Logging level set to {args.log_level.upper()}")

    try:
        # 1. Initialize components
        actual_data_path = Path(args.data_path)
        if not actual_data_path.is_file():
            logger.error(f"Data file not found at specified path: {actual_data_path}")
            print(
                f"Error: Data file not found at {actual_data_path}."
                f" Please check the --data_path argument."
            )
            return 1

        data_processor = CsvDataProcessor(data_path=str(actual_data_path))
        translator = LLMTranslator(device=args.device)
        classifier = LLMClassifier(device=args.device)
        query_parser = QueryParser()

        # 2. Create Orchestrator
        orchestrator = FreelancerQueryOrchestrator(
            data_processor=data_processor,
            classifier=classifier,
            translator=translator,
            parser=query_parser,
        )
        logger.info("All components initialized successfully.")

        # 3. Handle CLI arguments
        if args.test:
            run_test_queries(orchestrator)
        elif args.query:
            logger.info(
                f"Processing single query from CLI: '{args.query}'"
                f" (lang: {args.lang or 'auto-detect'})"
            )
            result = orchestrator.process_query(
                args.query, source_language_code=args.lang
            )
            print(result)
        else:
            run_interactive_mode(orchestrator)

    except Exception as e:
        logger.exception(
            "A critical error occurred during application setup or execution."
        )
        print(f"A critical error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
