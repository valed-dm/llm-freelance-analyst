# LLM Freelancer Analytics CLI
<!-- Badge Definitions -->
[python-badge]: https://img.shields.io/badge/python-3.12%2B-blue
[python-link]: https://www.python.org/
[pandas-badge]: https://img.shields.io/badge/pandas-2.2.3%2B-blue
[pandas-link]: https://pandas.pydata.org/
[transformers-badge]: https://img.shields.io/badge/transformers-4.52.4%2B-blue
[transformers-link]: https://huggingface.co/docs/transformers/index
[pytorch-badge]: https://img.shields.io/badge/PyTorch-2.2.2%2B-blue
[pytorch-link]: https://pytorch.org/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-link]: https://opensource.org/licenses/MIT

[![Python Version][python-badge]][python-link]
[![Pandas][pandas-badge]][pandas-link]
[![Transformers][transformers-badge]][transformers-link]
[![PyTorch][pytorch-badge]][pytorch-link]
[![Ruff][ruff-badge]][ruff-link]
[![License: MIT][license-badge]][license-link]

A command-line interface (CLI) tool for analyzing freelancer earnings data using natural language queries.
Our approach integrates a hybrid system: Natural Language Processing (NLP) for query understanding
and rule-based data processing for analysis. User queries are first interpreted by Language Models (LLMs)
for intent classification and multilingual translation. Subsequently, specific parameters are extracted
from the query using regex. These parameters drive precise analytical operations on the structured
dataset (CSV) via Pandas, ensuring both flexibility in input and accuracy in results without directly
exposing the dataset to the LLMs.


Run from Project Root: You should run this main.py using the module execution flag from your project's root directory (EORA):
```
python -m src.main --test
python -m src.main "your query here" --lang en
python -m src.main # For interactive mode
```

## Features

- **Natural Language Queries:** Ask questions about freelancer data in plain English or Russian.
- **Multilingual Support:** Input queries in Russian (translated to English for analysis).
- **Core Analytical Capabilities:**
  - **Income Comparison:** Compare average incomes between different freelancer groups (e.g., based on payment method, expertise).
  - **Income Distribution:** Show income statistics (mean, median, count) grouped by categories like region or payment method.
  - **Percentage Calculations:** Calculate percentages of freelancers meeting specific criteria (e.g., "percentage of experts with less than 10 projects").
  - **Average Income:** Determine the average income for a specific group of freelancers.
- **Modular Design:** Built with a clear separation of concerns for data processing, LLM interaction, query parsing, and orchestration.
- **Interactive Mode:** Engage in a conversation-like session to ask multiple queries.
- **Single Query Mode:** Get a quick answer to a single query passed as a command-line argument.
- **Test Mode:** Run a predefined suite of test queries to verify functionality.

## Technologies Used

- **Python:** Core programming language.
- **Pandas:** For efficient data manipulation and analysis of the CSV dataset.
- **Hugging Face `transformers`:** To leverage pre-trained language models for:
  - **Query Classification:** Using models like `google/flan-t5-small` to understand the intent of the user's query.
  - **Translation:** Using models like `Helsinki-NLP/opus-mt-ru-en` to translate Russian queries to English.
- **PyTorch:** As the backend deep learning framework for the Hugging Face models.
- **`safetensors`:** For secure and efficient model weight loading.
- **`sentencepiece`:** Tokenization library required by some translation models.
- **`argparse`:** For parsing command-line arguments.
- **`pathlib`:** For robust path manipulations.
- **`logging`:** For application logging.

## Project Summary

*   **Modular Application Structure:**
    *   `CsvDataProcessor`: Handles all data loading, cleaning, and direct analytical operations.
    *   `LLMClassifier`: Manages query classification using a language model.
    *   `LLMTranslator`: Provides translation capabilities for multilingual input.
    *   `QueryParser`: Extracts specific parameters from classified queries using regex.
    *   `FreelancerQueryOrchestrator`: Coordinates the overall query processing workflow.

*   **Core Analytical Capabilities:**
    The system can proficiently handle a variety of analytical queries:
    *   **Income Comparison:** Compares average incomes between different freelancer groups, including contextual handling for "X vs. Other" scenarios (e.g., "crypto vs. non-crypto").
    *   **Income Distribution:** Shows income statistics (mean, median, count) grouped by specified categories.
    *   **Percentage Calculations:** Calculates percentages of freelancers meeting defined conditions and criteria, with improved parsing for query variations.
    *   **Average Income Calculation:** Determines the average income for specified single groups of freelancers.
    *   **Mode Calculation:** Identifies the most common (modal) value within a given category (e.g., "most common job category").

*   **Multilingual Input:**
    Supports queries in Russian, which are automatically translated into English before analysis, broadening the tool's accessibility.

*   **Enhanced Robustness:**
    Development has addressed and resolved several complex technical challenges, including:
    *   Nuances in LLM interaction and prompt engineering for accurate query understanding.
    *   Secure and efficient model loading using `safetensors`.
    *   Management of critical dependencies like `sentencepiece`.
    *   Refined regular expression parsing for diverse query phrasings.

*   **Comprehensive Logging:**
    Informative logging is integrated throughout the application, providing clear insights into the execution flow and aiding in debugging or monitoring.

*   **Functional CLI Interface:**
    Offers a user-friendly command-line interface with multiple modes of operation:
    *   Interactive session for sequential queries.
    *   Single-query execution via command-line arguments.
    *   A test mode to run a predefined suite of queries for validation.


## Краткая характеристика примененного решения:

Для выполнения задачи выбран стандартный подход: LLM (Large Language Model (artificial intelligence))
для распознания общего уровня и смысла запроса, перевода с одного языка на другой, и последующий парсинг
строк для точного получения необходимых параметров. Такой сценарий называют "Гибридная система распознавания языка"
("HYBRID NLU SYSTEM").

Перевод запроса с русского на английский язык применен, потому что доля английского языка при обучении
модели преобладает, соответственно точность распознавания/релевантности ответов в этом случае выше.

NLU = (Natural Language Understanding),
NLP = (Natural Language Processing )

Плюсы решения:
Используем языковую модель для распознавания ЦЕЛИ запроса. Таким образом пользовательский ввод не требует
использования заранее заданных команд для управления программой. После того как цель запроса выявлена
регулярные выражения и логика позволяют осуществлять контроль над получением точных параметров для выполнения операции.

Одним из важных моментов является простота, скорость и дешевизна использования локальных легковесных
языковых моделей для распознавания/перевода по сравнению с API-based моделями. Можно начать с небольшого количества
запросов и постепенно углубляться расширяя правила для модели (LLM prompt), нормализацию, парсер, обработчик данных.

[<img src="docs/images/img_01.png" width="1200"/>]()

[<img src="docs/images/img_02.png" width="1200"/>]()

[<img src="docs/images/img_03.png" width="1200"/>]()

[<img src="docs/images/img_04.png" width="1200"/>]()

[<img src="docs/images/img_05.png" width="600"/>]()

[<img src="docs/images/img_06.png" width="600"/>]()

## Оценка эффективности решения

Эта задача представляется не такой простой, как кажется на первый взгляд.
Если заняться ей с нуля, то очевидно выйдем за рамки тестового задания, в ом числе и по срокам.
Коротко о сценариях:

Evaluating the efficiency and accuracy of your hybrid NLP and data analysis system requires a multi-faceted approach, looking at both the NLP components and the final analytical output. Here's a breakdown of how you can do it:

I. Evaluating NLP Components (Translation & Classification)

    A. LLM-Powered Translation Accuracy:

        Method:

            Create a small, representative dataset of Russian queries.

            Manually translate these queries into high-quality English (this is your "gold standard" or ground truth).

            Run your Russian queries through the LLMTranslator.

            Compare the machine-translated output against your gold-standard translations.

        Metrics:

            BLEU Score: Commonly used for machine translation evaluation. It measures n-gram overlap between machine translation and reference translations. (Requires libraries like nltk or sacrebleu).

            METEOR, TER (Translation Edit Rate): Other automated metrics.

            Qualitative Human Evaluation: Have a bilingual speaker rate the fluency, adequacy, and meaning preservation of the translations. This is often more insightful than automated metrics alone for specific domains. Does the translation retain the intent of the original query for analytical purposes?

    B. LLM-Powered Query Classification Accuracy:

        Method:

            Create a diverse test set of queries (both English and translated Russian).

            For each query, manually label its correct classification type (e.g., income_comparison, percentage_calculation, list_unique_values, other). This is your ground truth.

            Run these queries through your LLMClassifier (and subsequent normalize_classification).

            Compare the system's classification with your ground truth labels.

        Metrics:

            Accuracy: (Number of correctly classified queries) / (Total number of queries).

            Precision, Recall, F1-Score (per class): Especially important if some query types are more critical or if the dataset is imbalanced.

                Precision (for "income_comparison"): Of all queries classified as "income_comparison" by the system, how many actually were?

                Recall (for "income_comparison"): Of all actual "income_comparison" queries, how many did the system correctly identify?

            Confusion Matrix: Helps visualize which categories are being confused with others (e.g., is average_income_calculation often mistaken for income_comparison?).

II. Evaluating Parameter Parsing (QueryParser)

    Method:

        Using the same (or an expanded) test set from classification, for each query that was correctly classified, manually identify the correct parameters that should be extracted.

            E.g., for "Compare income of PayPal users vs crypto users" (classified as income_comparison), correct params: group1_desc='paypal users', group2_desc='crypto users'.

            E.g., for "percentage of experts completed less than 10 projects" (classified as percentage_calculation), correct params: condition='experts', criteria='completed less than 10 projects'.

        Run the correctly classified queries through the appropriate QueryParser method.

        Compare the extracted parameters with your ground truth.

    Metrics:

        Exact Match Accuracy (per parameter type): For each parameter (e.g., group1_desc), what percentage of the time was it extracted exactly correctly?

        Slot Filling Metrics (Precision, Recall, F1 for each slot): Similar to classification metrics, but for individual parameters.

        Qualitative Error Analysis: When parsing fails or is incorrect, why? Is the regex too strict/loose? Is the query phrasing too ambiguous for the current regexes?

III. Evaluating End-to-End Analytical Accuracy (CsvDataProcessor & Orchestrator)

    Method:

        For a subset of your test queries (covering all supported analytical types), manually calculate the expected correct answer from your freelancer_earnings_bd.csv dataset. This is your ground truth for the final output.

        Run these queries through the entire system (FreelancerQueryOrchestrator.process_query).

        Compare the system's final output string (the numerical result or textual answer) with your manually calculated ground truth.

    Metrics:

        Exact Match Accuracy (for numerical results): Does the number in the system's output match the ground truth number? (Allow for minor floating-point differences).

        Semantic Correctness (for textual parts): Does the textual description in the output (e.g., "experts", "non-crypto") accurately reflect the query and the data used?

        Overall Task Success Rate: What percentage of queries yield the completely correct analytical answer?

IV. Evaluating System Efficiency (Performance)

    Method:

        Time the execution of different parts of the system for a representative set of queries.

            Translation time.

            Classification time.

            Parsing time (usually negligible compared to LLM calls).

            Data processing time (Pandas operations).

            Total end-to-end query processing time.

        Use Python's time module or timeit for measurements. Your existing logging with timestamps is already a good start.

    Metrics:

        Average Latency (per query type / overall): How long does it take to get an answer?

        Throughput (if applicable for batch processing, less so for CLI): How many queries can be processed per unit of time?

        Resource Usage: Monitor CPU, memory (RAM), and GPU (VRAM if using GPU for LLMs) consumption during operation, especially for the LLM loading and inference steps. Tools like htop, Activity Monitor, nvidia-smi.

Practical Steps for Evaluation:

    Create a Test Suite: Your test_queries list in cli.py is a good starting point. Expand it to be comprehensive, covering:

        All supported analytical types.

        Variations in phrasing for each type.

        Queries in different supported languages (Russian).

        Edge cases (empty results, unparseable components).

        Queries that should be classified as "other."

    Establish Ground Truth: This is the most labor-intensive part. You need:

        Correct translations.

        Correct classifications.

        Correctly parsed parameters.

        Correct final analytical answers.

    Automate Where Possible:

        Write scripts to run your test suite and collect system outputs.

        Use libraries for automated metrics like BLEU, accuracy, precision/recall/F1.

    Iterate: Evaluation is not a one-time thing. As you make changes, add features, or refine prompts, re-evaluate to ensure you're not introducing regressions and that improvements are effective.

    Focus on Failure Cases: Pay close attention to queries where the system fails. Error analysis will guide your improvements (e.g., prompt tuning, regex adjustments, new data processing logic).

By systematically evaluating these different aspects, you'll get a clear picture of your system's strengths and weaknesses, guiding further development and refinement.
