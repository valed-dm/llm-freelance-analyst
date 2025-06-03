from transformers import pipeline


classifier_nlp = pipeline("text2text-generation", model="google/flan-t5-small")

prompt_template_to_test = """Classify this freelancer data query into one of these types.
    If the query asks for a direct comparison of income between two groups, use 'income_comparison'.
    If the query asks for how income is spread across categories, use 'income_distribution'.
    If the query explicitly asks for a 'percentage' of a group meeting a condition, use 'percentage_calculation'.
    If the query asks for an 'average income' or 'mean income' of a single group, use 'average_income_calculation'.
    If the query asks to find the most frequent or common item in a category, use 'mode_calculation'.
    If the query asks to list all unique values, distinct entries, or all available options in a specific category or column, use 'list_unique_values'.
    For all other types of queries, especially general information requests not covered by the above, use 'other'.

    - income_comparison (e.g., 'Compare income of X versus Y')
    - income_distribution (e.g., 'Show income distribution by region')
    - percentage_calculation (e.g., 'What percentage of experts did Z?')
    - average_income_calculation (e.g., 'What is the average income of group X?', 'Find the mean earnings for experts.')
    - mode_calculation (e.g., 'What is the most common job category?', 'Most frequent payment method')
    - list_unique_values (e.g., 'What are all job categories?', 'List all payment methods', 'Show distinct client regions.')
    - other (e.g., 'Tell me about freelancers', 'What is the highest salary?')

    Query: {query}"""  # noqa: E501

query = "What are all job categories?"
full_prompt = prompt_template_to_test.format(query=query)

print("--- PROMPT ---")
print(full_prompt)
print("--- END PROMPT ---")

output = classifier_nlp(full_prompt, max_new_tokens=20)
print("\n--- LLM RAW OUTPUT ---")
print(output[0]["generated_text"])
