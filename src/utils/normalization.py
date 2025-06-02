import logging


logger = logging.getLogger(__name__)


def normalize_classification(text: str) -> str:
    """
    Normalizes the raw classification output from the LLM to a standard category name.
    """
    if not text:
        logger.warning(
            "Received empty text for classification normalization."
            " Defaulting to 'other'."
        )
        return "other"

    text_lower = text.lower().strip()

    # 1. Check for exact internal keywords (most precise)
    # Ensure 'other' is checked early if the LLM is good at outputting it.
    known_categories = [
        "income_comparison",
        "income_distribution",
        "percentage_calculation",
        "average_income_calculation",
        "mode_calculation",
        "list_unique_values",
        "other",
    ]
    for cat in known_categories:
        if cat == text_lower:
            return cat
        # Also check for "category_name" in "some text category_name more text"
        # This handles cases where LLM might output more than just the keyword.
        # Example: if LLM outputs "classification is income_comparison",
        # this should catch it.
        if (
            f" {cat} " in f" {text_lower} "
            or text_lower.startswith(cat + " ")
            or text_lower.endswith(" " + cat)
            or cat.replace("_", " ") == text_lower
            or f" {cat.replace('_', ' ')} " in f" {text_lower} "
            or text_lower.startswith(cat.replace("_", " ") + " ")
            or text_lower.endswith(" " + cat.replace("_", " "))
        ):
            return cat

    # 2. Check for strong keyword indicators (if LLM uses slightly different phrasing)
    # These should be more specific than just partial matches of the category names.
    if "compare" in text_lower and "income" in text_lower:
        return "income_comparison"
    if "percentage" in text_lower and (
        "calculate" in text_lower
        or "what is the" in text_lower
        or "portion of" in text_lower
    ):
        return "percentage_calculation"
    if ("average" in text_lower or "mean" in text_lower) and "income" in text_lower:
        return "average_income_calculation"
    if "most common" in text_lower or "most frequent" in text_lower:
        return "mode_calculation"
    if (
        "list all" in text_lower
        or "what are all" in text_lower
        or "distinct" in text_lower
        or "unique values" in text_lower
    ):
        return "list_unique_values"

    # Specific check for "income distribution" - if "distribution" is present
    # AND "income"
    if "distribution" in text_lower and "income" in text_lower:
        return "income_distribution"
    # If "distribution" is present but NOT income, it's likely "other" or an error.
    # For "weather_distribution", this means it should NOT become "income_distribution"
    # here.
    # If it was just "distribution" from LLM, it's ambiguous. The prompt should guide
    # LLM better.

    # 3. If LLM indicates "other" with different common phrasings
    other_indicators = [
        "not applicable",
        "unknown query",
        "unsupported",
        "cannot classify",
        "general query",
    ]
    if any(indicator in text_lower for indicator in other_indicators):
        return "other"

    # here, no confident mapping was found.
    logger.warning(
        f"Could not confidently normalize LLM classification output: '{text}'. "
        f"Cleaned text for check: '{text_lower}'. Defaulting to 'other'."
    )
    return "other"
