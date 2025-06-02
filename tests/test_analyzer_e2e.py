import pytest


TEST_CASES = [
    (
        "Compare income of freelancers who use PayPal versus those who use crypto",
        "en",
        [
            "earns",
            "more",
            "than",
            "paypal",
            "crypto",
            "%",
        ],
    ),
    (
        "What percentage of experts completed less than 100 projects?",
        "en",
        ["%", "expert freelancers", "less than 100"],
    ),
    (
        "Show income distribution by region",
        "en",
        ["Income distribution by region", "mean", "median", "count", "Asia", "USA"],
    ),
    (
        "What is the average income of freelancers using mobile banking?",
        "en",
        ["average income", "mobile banking", "$"],
    ),
    (
        "What's the income difference between PayPal users and crypto users?",
        "en",
        ["paypal users", "crypto users", "$", "%", "earns", "more", "than"],
    ),
    (
        "Calculate the percentage of beginner freelancers who finished more than"
        " 10 projects.",
        "en",
        ["%", "beginner freelancers", "more than 10"],
    ),
    (
        "What portion of experts have completed exactly 0 projects?",
        "en",
        ["%", "expert freelancers", "exactly 0"],
    ),
    (
        "How is income distributed by payment method?",
        "en",
        [
            "Income distribution by payment_method",
            "PayPal",
            "Crypto",
        ],
    ),
    (
        "Find the mean income for experts.",
        "en",
        [
            "average income",
            "experts",
            "$",
        ],
    ),
    (
        "What is the most common job category?",
        "en",
        [
            "The most common job_category(s)",
            "Graphic Design",
        ],
    ),
    (
        "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 "
        "проектов?",
        "ru",
        [
            "%",
            "expert freelancers",
            "less than 100",
        ],
    ),
    (
        "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте,"
        " по сравнению с другими способами оплаты?",
        "ru",
        ["crypto", "other (non-crypto) payment methods", "%"],
    ),
    (
        "Как распределяется доход фрилансеров в зависимости от региона проживания?",
        "ru",
        ["Income distribution by region", "Asia", "USA"],
    ),
    (
        "Tell me about the weather today.",
        "en",
        ["I'm not sure how to answer"],  # Expected for "other" classification
    ),
    (
        "Average income.",
        "en",  # Should be unparseable for average_income
        ["Could not parse the group for average income calculation"],
    ),
]


@pytest.mark.parametrize("query_text, lang_code, expected_substrings", TEST_CASES)
def test_freelancer_query(orchestrator, query_text, lang_code, expected_substrings):
    """
    Tests the orchestrator's process_query method with various inputs.
    Asserts that the output contains expected substrings.
    """
    result = orchestrator.process_query(query_text, source_language_code=lang_code)

    assert result is not None, "Orchestrator returned None, expected a string response."
    assert isinstance(result, str), f"Expected string result, got {type(result)}"
    assert len(result.strip()) > 0, "Orchestrator returned an empty string."

    for substring in expected_substrings:
        assert substring.lower() in result.lower(), (
            f"Expected substring '{substring}' not found in result:\n'{result}'\n"
            f"for query:\n'{query_text}'"
        )
