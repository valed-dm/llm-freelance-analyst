import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


class QueryParser:
    """
    Parses classified a query text to extract specific parameters required for analysis
    using regular expressions.
    """

    def __init__(self) -> None:
        """Initializes the QueryParser."""
        logger.info("QueryParser initialized.")

    @staticmethod
    def parse_income_comparison(query_text: str) -> dict[str, Any] | None:
        """
        Parses an 'income_comparison' query to extract descriptions for two groups.

        Args:
            query_text (str):
                The query text (assumed to be in English and lowercased by caller).

        Returns:
            dict | None:
                A dictionary with 'group1_desc' and 'group2_desc' if parsing
                is successful, otherwise None or a dictionary with an error.
            Example: {'group1_desc': 'paypal users', 'group2_desc': 'crypto users'}
        """
        query_lower = query_text.lower().rstrip("?.!")
        match = None

        # Pattern 0: Specifically for "difference between X and Y"
        # Looks for "between", captures everything until "and",
        # then captures everything after "and".
        # Example: "what's the income difference between
        # [paypal users] and [crypto users]"
        # Example: "difference between [group a] and [group b]"
        if (
            "between" in query_lower and " and " in query_lower
        ):  # Quick check to see if this pattern is even relevant
            match_between_and = re.search(
                r"(?:difference|diff)\s+(?:in\s+|of\s+)?(?:income\s+)?between\s+(.*?)\s+and\s+(.*)",
                query_lower,
            )
            if not match_between_and:  # A simpler version if the above is too specific
                match_between_and = re.search(
                    r"between\s+(.*?)\s+and\s+(.*)", query_lower
                )
            if match_between_and:  # If either "between...and" pattern matched
                match = match_between_and  # Use this match

        if not match:
            # Pattern 1: "income of (freelancers) [G1] compared to/versus/than [G2]"
            match = re.search(
                r"income of (?:freelancers\s+)?(.*?)\s+(?:versus|vs\.?|"
                r"compared to|than)\s+(.*)",
                query_lower,
            )

        if not match:
            # Pattern 2: General "[G1] compared to/versus/than [G2]"
            match = re.search(
                r"(.*?)\s+(?:versus|vs\.?|compared to|than)\s+(.*)", query_lower
            )

        if not match:
            # Pattern 3: "who [G1] compared to/versus/than (those who use...) [G2]"
            match = re.search(
                r"who\s+(.*?)\s+(?:versus|vs\.?|compared to|than)\s+"
                r"(?:those who use |those using |those with |the |their )?(.*)",
                query_lower,
            )

        if match:
            group1_desc, group2_desc = map(str.strip, match.groups())
            logger.debug(
                f"QueryParser (income_comparison): Extracted Group 1: '{group1_desc}',"
                f" Group 2: '{group2_desc}'"
            )
            return {"group1_desc": group1_desc, "group2_desc": group2_desc}
        else:
            logger.warning(
                f"QueryParser (income_comparison): Could not parse groups from query:"
                f" '{query_text}'"
            )
            return None

    @staticmethod
    def parse_percentage_calculation(query_text: str) -> dict[str, Any] | None:
        """
        Parses a 'percentage_calculation' query to extract condition and criteria.
        Handles "percentage of" and "portion of" phrasing.
        """
        query_lower = query_text.lower().rstrip("?.!")

        condition_str = None
        criteria_str = None
        simple_match = None
        m_expert_in_criteria = None

        # Regex to handle "percentage of..." or "portion of..."
        # (?:percentage|portion) - non-capturing group for either word
        # of\s+ - matches " of "
        # (?P<condition>.*?) - named group for condition (non-greedy)
        # \s+ - space separator
        # (?P<criteria>(?:completed|finished|have|has|did|...)\s+.*|...)
        # - named group for criteria (your existing complex criteria part)

        # Main pattern attempt
        # This pattern tries to identify the subject (condition) and
        # the predicate (criteria)
        # It looks for common verbs or comparators to split.
        pattern_prefix = r"(?:percentage|portion)\s+of\s+"  # Common prefix for both
        refined_match = re.search(
            pattern_prefix
            + r"(?P<condition>.*?)\s+(?P<criteria>(?:completed|finished|have|has|did|"
            r"use|uses|using|are|is)\s+.*|(?:less|more|exactly|under|over)\s+(?:than\s+)?\d+.*)",
            query_lower,
        )

        if refined_match:
            condition_str = refined_match.group("condition").strip()
            criteria_str = refined_match.group("criteria").strip()
        else:
            simple_match = re.search(
                pattern_prefix + r"(?P<condition>.*?)\s+(?P<criteria>.*)", query_lower
            )
            if simple_match:
                condition_str = simple_match.group("condition").strip()
                criteria_str = simple_match.group("criteria").strip()

                # Optional: Post-correction for the simple match
                m_expert_in_criteria = re.search(
                    r"(who consider themselves experts|experts|beginners|advanced)",
                    criteria_str,
                    re.IGNORECASE,
                )
                generic_conditions = ["freelancers", "users", "people", "them", "those"]
                is_condition_generic = any(
                    gc_word == condition_str.lower() for gc_word in generic_conditions
                )

                if (
                    is_condition_generic
                    and m_expert_in_criteria
                    and len(condition_str.split()) < 3
                ):  # Heuristic: if condition is short
                    expert_phrase = m_expert_in_criteria.group(1)

                    # Avoid adding if already largely present
                    if expert_phrase.lower() not in condition_str.lower():
                        condition_str = f"{condition_str} {expert_phrase}"
                    elif (
                        condition_str.lower() == "freelancers"
                        and "expert" in expert_phrase.lower()
                    ):  # e.g. condition="freelancers", expert_phrase="experts"
                        condition_str = expert_phrase  # Prioritize the more specific

                    criteria_str = re.sub(
                        r"^\s*" + re.escape(expert_phrase) + r"\s*",
                        "",
                        criteria_str,
                        flags=re.IGNORECASE,
                    ).strip()

        if condition_str is not None and criteria_str is not None:
            if (
                not criteria_str and simple_match and m_expert_in_criteria
            ):  # If criteria became empty due to post-correction
                logger.warning(
                    f"QueryParser (percentage_calculation): Criteria became empty after"
                    f" post-correction for query: '{query_text}'. Original condition:"
                    f" '{simple_match.group('condition')}', Original criteria:"
                    f" '{simple_match.group('criteria')}'"
                )

            logger.debug(
                f"QueryParser (percentage_calculation): Condition: '{condition_str}',"
                f" Criteria: '{criteria_str}'"
            )
            return {"condition": condition_str, "criteria": criteria_str}
        else:
            logger.warning(
                f"QueryParser (percentage_calculation): Could not parse"
                f" condition/criteria from query: '{query_text}'"
            )
            return None

    @staticmethod
    def parse_average_income(query_text: str) -> dict[str, Any] | None:
        """
        Parses an 'average_income_calculation' query to extract the group description.

        Args:
            query_text (str): The query text (assumed to be in English and lowercased by caller).

        Returns:
            dict | None: A dictionary with 'group_description' if successful, else None.
        """  # noqa: E501
        query_lower = query_text.lower().rstrip("?.!")  # Normalize

        # Pattern: "average/mean income of/for [GROUP DESCRIPTION]"
        match = re.search(r"(?:average|mean) income (?:of|for)\s+(.*)", query_lower)
        if match:
            group_description = match.group(1).strip()
            logger.debug(
                f"QueryParser (average_income): Group Description:"
                f" '{group_description}'"
            )
            return {"group_description": group_description}
        else:
            logger.warning(
                f"QueryParser (average_income): Could not parse group description"
                f" from query: '{query_text}'"
            )
            return None

    @staticmethod
    def parse_income_distribution(query_text: str) -> dict[str, Any] | None:
        """
        Parses an 'income_distribution' query to extract the distribution column.
        Currently, very simple and assumes 'by region' or 'by payment_method' if mentioned.

        Args:
            query_text (str): The query text (assumed to be in English and lowercased by caller).

        Returns:
            dict | None: A dictionary with 'by_column' if successful, else None.
                         Defaults to 'region' if no specific column is parsed.
        """  # noqa: E501
        query_lower = query_text.lower().rstrip("?.!")  # Normalize
        by_column = "region"  # Default distribution column

        if (
            "by payment method" in query_lower
            or "distributed by payment method" in query_lower
        ):
            by_column = "payment_method"
        elif "by region" in query_lower or "distributed by region" in query_lower:
            by_column = "region"
        # add more specific parsing for other columns if needed

        logger.debug(
            f"QueryParser (income_distribution): Distribute by column: '{by_column}'"
            f" (derived from query: '{query_text}')"
        )
        return {"by_column": by_column}

    @staticmethod
    def parse_mode_calculation(query_text: str) -> dict[str, Any] | None:
        query_lower = query_text.lower().strip().rstrip("?.!")
        # Map common query phrases to DataFrame column names (cleaned or original based
        # on CsvDataProcessor)
        column_phrase_map = {
            "job category": "job_category",
            "payment method": "payment_method",
            "region": "region",
            "experience level": "expert_status",  # Note: expert_status is binary
            # to Add more mappings as needed
        }

        # Regex to find "most common/frequent X"
        match = re.search(
            r"(?:most\s+common|most\s+frequent|commonest)\s+(.*)", query_lower
        )
        if match:
            target_phrase = match.group(1).strip()
            for phrase, col_name in column_phrase_map.items():
                if phrase in target_phrase:  # Simple substring check
                    logger.debug(
                        f"QueryParser (mode_calculation): Parsed column '{col_name}'"
                        f" from phrase '{target_phrase}'"
                    )
                    return {"column_name": col_name}
            logger.warning(
                f"QueryParser (mode_calculation): Could not map phrase"
                f" '{target_phrase}' to a known column."
            )
        else:
            logger.warning(
                f"QueryParser (mode_calculation): Could not parse target for mode"
                f" from query: '{query_text}'"
            )
        return None

    @staticmethod
    def parse_list_unique_values(query_text: str) -> dict[str, Any] | None:
        query_lower = query_text.lower().strip().rstrip("?.!")
        # Map query phrases to DataFrame column names (cleaned names you use internally)
        column_phrase_map = {
            "job categories": "job_category",
            "platforms": "platform",
            "experience levels": "expert_status",  # This is binary (0/1)
            "client regions": "region",
            "payment methods": "payment_method",
            "project types": "project_type",
            # Add singular forms too
            "job category": "job_category",
            "platform": "platform",
            "experience level": "expert_status",
            "client region": "region",
            "payment method": "payment_method",
            "project type": "project_type",
        }

        match = re.search(
            r"(?:what\s+are\s+all|list\s+all|show\s+all|show\s+me\s+the|"
            r"distinct|unique)\s+(.*)",
            query_lower,
        )
        if match:
            target_phrase = match.group(1).strip()
            for phrase, col_name in column_phrase_map.items():
                if phrase in target_phrase:
                    logger.debug(
                        f"QueryParser (list_unique): Parsed column"
                        f" '{col_name}' from phrase '{target_phrase}'"
                    )
                    return {"column_name": col_name}
            logger.warning(
                f"QueryParser (list_unique): Could not map phrase"
                f" '{target_phrase}' to a known column for listing"
                f" unique values."
            )
        else:
            logger.warning(
                f"QueryParser (list_unique): Could not parse target column"
                f" from query: '{query_text}'"
            )
        return None


# Example usage for testing this class in isolation:
if __name__ == "__main__":
    parser = QueryParser()

    print("\n--- Testing Income Comparison ---")
    test_ic_1 = "Compare income of freelancers who use PayPal versus those who use crypto"  # noqa: E501
    print(f"Input: {test_ic_1}\nOutput: {parser.parse_income_comparison(test_ic_1)}")
    test_ic_2 = "income of experts compared to beginners"
    print(f"Input: {test_ic_2}\nOutput: {parser.parse_income_comparison(test_ic_2)}")
    test_ic_3 = "how much do crypto users make than paypal ones?"  # "ones" is tricky
    print(f"Input: {test_ic_3}\nOutput: {parser.parse_income_comparison(test_ic_3)}")

    print("\n--- Testing Percentage Calculation ---")
    test_pc_1 = (
        "What percentage of freelancers who consider themselves experts have"
        " completed less than 100 projects?"
    )
    print(
        f"Input: {test_pc_1}\nOutput: {parser.parse_percentage_calculation(test_pc_1)}"
    )
    test_pc_2 = "percentage of beginners with more than 5 projects"
    print(
        f"Input: {test_pc_2}\nOutput: {parser.parse_percentage_calculation(test_pc_2)}"
    )
    # Condition is just "freelancers"
    test_pc_3 = "percentage of freelancers completed exactly 0 projects"
    print(
        f"Input: {test_pc_3}\nOutput: {parser.parse_percentage_calculation(test_pc_3)}"
    )

    print("\n--- Testing Average Income ---")
    test_ai_1 = "What is the average income of freelancers using mobile banking?"
    print(f"Input: {test_ai_1}\nOutput: {parser.parse_average_income(test_ai_1)}")
    test_ai_2 = "mean income for experts"
    print(f"Input: {test_ai_2}\nOutput: {parser.parse_average_income(test_ai_2)}")

    print("\n--- Testing Income Distribution ---")
    test_id_1 = "Show income distribution by region"
    print(f"Input: {test_id_1}\nOutput: {parser.parse_income_distribution(test_id_1)}")
    test_id_2 = "How is income distributed by payment method?"
    print(f"Input: {test_id_2}\nOutput: {parser.parse_income_distribution(test_id_2)}")
    test_id_3 = "income distribution"  # Should default to region
    print(f"Input: {test_id_3}\nOutput: {parser.parse_income_distribution(test_id_3)}")
