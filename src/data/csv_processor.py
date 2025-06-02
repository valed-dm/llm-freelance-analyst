import logging
import re

import pandas as pd


logger = logging.getLogger(__name__)


class CsvDataProcessor:
    """
    Handles loading, cleaning, and performing analytical operations on freelancer data
    loaded from a CSV file into a Pandas DataFrame.
    """

    def __init__(self, data_path: str) -> None:
        """Loads data, calls _clean_data

        Args:
            data_path: Path to CSV file to load
        """
        self.data: pd.DataFrame | None = None

        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
        except FileNotFoundError:
            logger.exception(f"Error: Data file not found at {data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.exception(f"Error: Data file at {data_path} is empty.")
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while loading data from {data_path}: {e}"
            )
            raise
        self._clean_data()

    def _clean_data(self) -> None:
        """Performs comprehensive cleaning of the loaded DataFrame.

        Executes the following cleaning operations in sequence:
        1. Column Renaming: Standardizes column names using predefined mapping
        2. Missing Value Handling: Fills NA values with sensible defaults
        3. Expert Status Conversion: Normalizes expertise levels to binary values

        Column Renaming:
        - Maps original column names to standardized versions
        - Only renames columns that exist in the DataFrame
        - Logs warnings for any missing expected columns

        Missing Value Handling:
        - Numeric columns filled with median/0
        - Categorical columns filled with mode/default strings
        - Handles cases where target columns don't exist

        Expert Status Conversion:
        - Converts string values to binary (0/1) representation:
            - 1 for "expert" or "advanced"
            - 0 for all other values (including missing)
        - Preserves original values if the column doesn't exist

        Error Handling:
        - Returns immediately if no data is loaded
        - Logs warnings for all non-critical issues
        - Maintains original data for columns not specified

        Side Effects:
        - Modifies self.data in place
        - Generates INFO/WARNING level log messages
        """
        if self.data is None:
            logger.error("Cannot clean data: DataFrame not loaded.")
            return

        column_mapping = {
            "Earnings_USD": "income",
            "Experience_Level": "expert_status",
            "Job_Completed": "projects_completed",
            "Client_Region": "region",
            "Payment_Method": "payment_method",
            "Job_Category": "job_category",
        }
        # Only rename columns that exist to avoid KeyErrors if CSV is different
        existing_cols_to_rename = {
            k: v for k, v in column_mapping.items() if k in self.data.columns
        }
        self.data.rename(columns=existing_cols_to_rename, inplace=True)
        if len(existing_cols_to_rename) < len(column_mapping):
            missing_original_cols = set(column_mapping.keys()) - set(
                existing_cols_to_rename.keys()
            )
            logger.warning(
                f"Could not rename some columns as they were not found:"
                f" {missing_original_cols}"
            )
        logger.info(f"Renamed DataFrame columns: {existing_cols_to_rename}")

        if "income" in self.data.columns:
            income_median = self.data["income"].median()
        else:
            logger.warning(
                "'income' column not found. Defaulting median income to 0 for fillna."
            )
            income_median = 0

        fill_values = {
            "income": income_median,
            "projects_completed": 0,
            "expert_status": "Beginner",
        }
        existing_cols_to_fill = {
            k: v for k, v in fill_values.items() if k in self.data.columns
        }
        self.data.fillna(existing_cols_to_fill, inplace=True)
        logger.info(f"Filled missing values using: {existing_cols_to_fill}")

        if "expert_status" in self.data.columns:
            self.data["expert_status"] = self.data["expert_status"].apply(
                lambda x: (
                    1 if str(x).lower().strip() in ["expert", "advanced"] else 0
                )  # Added strip()
            )
            logger.info(
                "Converted 'expert_status' to binary"
                " (1 for expert/advanced, 0 otherwise)."
            )
        else:
            logger.warning(
                "'expert_status' column not found. Cannot convert to binary."
            )
        logger.info("Data cleaning process completed.")

    def _create_group_filter(self, description: str) -> pd.Series | None:
        """Creates a boolean filter Series based on the group description.

        Parses natural language descriptions to create pandas Series filters for
        grouping data. Handles both positive and negative (non-) conditions.

        Args:
            description: Natural language description of the group filter.
                Examples:
                - "crypto" → filters for cryptocurrency payments
                - "non-expert" → filters for non-expert users
                - "mobile banking" → filters for mobile payment methods

        Returns:
            pd.Series | None: Boolean mask Series suitable for DataFrame filtering,
                or None if:
                - DataFrame not loaded
                - Empty description provided
                - Required columns missing
                - Unrecognized description pattern

        Supported Patterns:
            Payment Methods:
            - "crypto"/"cryptocurrency" → payment_method contains crypto
            - "paypal" → payment_method contains PayPal
            - "mobile banking" → payment_method contains mobile
            - "non-{method}" → negation of above

            Expertise Levels:
            - "expert"/"advanced" → expert_status == 1
            - "beginner" → expert_status == 0
            - "non-expert"/"non-beginner" → negation of above

        Notes:
            - Case insensitive matching
            - Returns None for any unrecognized patterns
            - Logs warnings for unprocessable descriptions
        """
        if self.data is None:
            logger.error("Cannot create group filter: DataFrame not loaded.")
            return None

        desc_lower = description.lower().strip()
        if not desc_lower:  # Handle empty description
            logger.warning("Empty group description provided for filter creation.")
            return None

        # Handle negation first
        if desc_lower.startswith("non-"):
            actual_method_or_status = desc_lower.split("non-", 1)[1].strip()
            if actual_method_or_status in ["crypto", "cryptocurrency", "crypts"]:
                if "payment_method" in self.data.columns:
                    return ~self.data["payment_method"].str.contains(
                        "crypto", case=False, na=False
                    )
            elif actual_method_or_status == "paypal":
                if "payment_method" in self.data.columns:
                    return ~self.data["payment_method"].str.contains(
                        "paypal", case=False, na=False
                    )
            elif (
                actual_method_or_status == "mobile"
                or actual_method_or_status == "mobile banking"
            ):
                if "payment_method" in self.data.columns:
                    return ~self.data["payment_method"].str.contains(
                        "mobile", case=False, na=False
                    )
            elif actual_method_or_status in ["expert", "advanced"]:
                if "expert_status" in self.data.columns:
                    return ~(self.data["expert_status"] == 1)
            elif actual_method_or_status == "beginner":
                if "expert_status" in self.data.columns:
                    return ~(self.data["expert_status"] == 0)
            logger.warning(
                f"Unrecognized 'non-' type for filter: {actual_method_or_status}"
            )
            return None  # Unrecognized "non-" type or required column missing

        # Handle positive conditions
        if "payment_method" in self.data.columns:
            if (
                "cryptocurrency" in desc_lower
                or "crypto" in desc_lower
                or "crypts" in desc_lower
            ):
                return self.data["payment_method"].str.contains(
                    "crypto", case=False, na=False
                )
            elif "mobile banking" in desc_lower:
                return self.data["payment_method"].str.contains(
                    "mobile", case=False, na=False
                )
            elif "paypal" in desc_lower:
                return self.data["payment_method"].str.contains(
                    "paypal", case=False, na=False
                )

        if "expert_status" in self.data.columns:
            if "expert" in desc_lower or "advanced" in desc_lower:
                return self.data["expert_status"] == 1
            elif "beginner" in desc_lower:
                return self.data["expert_status"] == 0

        logger.warning(
            f"Could not parse group description: '{description}' into a known filter."
        )
        return None

    def compare_income(self, group1_desc: str, group2_desc: str) -> str:
        """Compares average income between two freelancer groups
        with natural language parsing."""
        # Validate data and initialize filters
        if self.data is None:
            return "Error: Data not loaded for income comparison."

        filter1, filter2, payment_type = self._initialize_filters(
            group1_desc, group2_desc
        )

        # Validate groups
        validation_result = self._validate_groups(
            filter1, filter2, group1_desc, group2_desc, payment_type
        )
        if validation_result is not None:
            return validation_result

        # Calculate and compare incomes
        return self._calculate_comparison(
            filter1, filter2, group1_desc, group2_desc, payment_type
        )

    def _initialize_filters(self, group1_desc: str, group2_desc: str) -> tuple:
        """Initialize and potentially modify filters based on group descriptions."""
        filter1 = self._create_group_filter(group1_desc)
        filter2 = self._create_group_filter(group2_desc)
        payment_type = None

        if (
            "payment_method" in self.data.columns
            and filter1 is not None
            and filter2 is None
        ):
            filter2, payment_type = self._handle_other_payment_methods(
                group1_desc, group2_desc
            )

        return filter1, filter2, payment_type

    def _handle_other_payment_methods(
        self, group1_desc: str, group2_desc: str
    ) -> tuple:
        """Handle a special case of 'other payment methods'."""
        g2_desc_lower = group2_desc.lower()
        is_other_payment = any(
            phrase in g2_desc_lower
            for phrase in [
                "other way",
                "other payment",
                "another payment",
                "different payment",
            ]
        )

        if not is_other_payment:
            return None, None

        logger.info(
            f"Interpreting '{group2_desc}' as 'other payment methods'"
            f" relative to '{group1_desc}'"
        )

        g1_desc_lower = group1_desc.lower()
        payment_types = {
            "crypto": ["crypto", "crypts"],
            "paypal": ["paypal"],
            "mobile banking": ["mobile banking"],
        }

        for payment_type, keywords in payment_types.items():
            if any(keyword in g1_desc_lower for keyword in keywords):
                filter2 = ~self.data["payment_method"].str.contains(
                    keywords[0], case=False, na=False
                )
                logger.info(
                    f"Interpreted '{group2_desc}' as 'non-{payment_type}'"
                    f" payment methods"
                )
                return filter2, payment_type

        logger.warning(
            f"Could not determine payment method for Group 1 ('{group1_desc}')"
        )
        return None, None

    def _validate_groups(
        self, filter1, filter2, group1_desc, group2_desc, payment_type
    ) -> str | None:
        """Validate that groups exist and have data."""
        if filter1 is None or filter2 is None:
            missing = []
            if filter1 is None:
                missing.append(f"group 1 ('{group1_desc}')")
            if filter2 is None:
                missing.append(f"group 2 ('{group2_desc}')")
            return f"Could not identify groups to compare: {', '.join(missing)}."

        group1_count = self.data.loc[filter1].shape[0]
        group2_count = self.data.loc[filter2].shape[0]

        if group1_count == 0 or group2_count == 0:
            empty = []
            if group1_count == 0:
                empty.append(f"group 1 ('{group1_desc}')")
            if group2_count == 0:
                desc = (
                    f"non-{payment_type} payment methods"
                    if payment_type
                    else group2_desc
                )
                empty.append(f"group 2 ('{desc}')")
            return ". ".join(empty) + "."

        return None

    def _calculate_comparison(
        self, filter1, filter2, group1_desc, group2_desc, payment_type
    ) -> str:
        """Calculate and format the comparison results."""
        mean1 = self.data.loc[filter1, "income"].mean()
        mean2 = self.data.loc[filter2, "income"].mean()

        if pd.isna(mean1) or pd.isna(mean2):
            return (
                "Could not calculate income for one or both groups "
                "(no valid income data)."
            )

        # Format group 2 description
        group2_output_desc = (
            f"other (non-{payment_type}) payment methods"
            if payment_type and filter2 is not None
            else group2_desc
        )

        # Handle zero income cases
        if abs(mean2) < 1e-9:
            if abs(mean1) < 1e-9:
                return (
                    f"Both groups '{group1_desc}' and '{group2_output_desc}'"
                    f" have $0.00 average income."
                )
            return (
                f"Group '{group1_desc}' averages ${mean1:,.2f}. "
                f"Group '{group2_output_desc}' has $0.00 average income."
            )

        # Calculate and format comparison
        diff = mean1 - mean2
        pct_diff = (diff / mean2) * 100

        return (
            f"Group '{group1_desc}' earns ${diff:,.2f} ({pct_diff:+.1f}%) "
            f"more than '{group2_output_desc}' (${mean1:,.2f} vs ${mean2:,.2f})."
        )

    def get_income_distribution(self, by_column: str) -> str:
        """Generates comprehensive income distribution statistics grouped by
          a specified column.

        Calculates and formats key income metrics (mean, median, count) for each
         category in the specified grouping column, with robust error handling and
         data validation.

        Args:
            by_column: Name of the column to group by (e.g., 'payment_method', 'region')
                      Column must exist in the DataFrame and contain at least some
                      non-NA values

        Returns:
            str: Formatted distribution results or error message containing either:
                 - Tabular statistics (mean, median, count) per group, or
                 - Specific error message if data requirements aren't met

        Statistics Calculated:
            - mean: Average income per group
            - median: Middle income value per group
            - count: Number of observations per group

        Error Handling:
            - Validates presence of required columns
            - Handles empty/NA groups (with warning)
            - Catches and reports calculation exceptions
            - Provides specific feedback for:
              * Missing data
              * Invalid columns
              * Calculation errors

        Edge Cases:
            - Returns warning if grouping column is all NA values
            - Handles empty DataFrames gracefully
            - Preserves NA groups in output (dropna=False)

        Example Output:
            Income distribution by payment_method:
                             mean  median  count
            payment_method
            crypto        5234.56 4200.00    125
            paypal        4123.45 3800.00    210
            mobile        3876.54 3500.00     85
            NaN           4012.34 3900.00     15

        Notes:
            - Includes NaN/None as a separate group if present
            - Uses median for robust central tendency measure
            - Maintains original column dtype for grouping
        """
        if self.data is None:
            return "Error: Data not loaded for income distribution."

        if by_column not in self.data.columns:
            logger.warning(f"Distribution column '{by_column}' not found in data.")
            return f"Cannot distribute by '{by_column}' - column not found."

        # Ensure 'income' column exists
        if "income" not in self.data.columns:
            logger.error("'income' column missing, cannot generate distribution.")
            return "Error: 'income' column is missing from the data."

        try:
            if self.data[by_column].isnull().all():
                logger.warning(
                    f"Column '{by_column}' contains all NaN values."
                    f" Distribution may be empty or misleading."
                )

            stats = self.data.groupby(by_column, dropna=False)["income"].agg(
                ["mean", "median", "count"]
            )  # dropna=False to include NaN groups
            if stats.empty:
                return (
                    f"No data available to generate income distribution by"
                    f" {by_column} (e.g., all values in '{by_column}'"
                    f" are NaN and 'income' is empty)."
                )
            return f"Income distribution by {by_column}:\n{stats.to_string()}"
        except Exception as e:
            logger.exception(
                f"Error calculating income distribution by '{by_column}': {e}"
            )
            return f"Error calculating income distribution by '{by_column}'."

    def calculate_percentage(self, condition: str, criteria: str) -> str:
        """Calculates and formats percentage statistics based on natural language
        conditions.

        Features:
        - Parses natural language conditions and criteria
        - Supports project count comparisons ("more than X projects")
        - Handles subgroup analysis with automatic filtering
        - Provides clear error messaging

        Args:
            condition: Filter condition for subgroup analysis
            (e.g., "experts", "crypto users")
                      Special values:
                      - "all" or empty: Use entire dataset
                      - "freelancers": Alias for all
            criteria: Calculation criteria in natural language format:
                     - "less than X projects"
                     - "more than X projects"
                     - "exactly X projects"

        Returns:
            str: Formatted result containing either:
                 - Percentage calculation with context
                 - Specific error message if requirements aren't met

        Supported Criteria:
            Project Count Comparisons:
            - "more than 5 projects"
            - "less than 10 projects"
            - "exactly 3 projects"

        Condition Handling:
            - Uses _create_group_filter for condition parsing
            - Maintains original dataset if the condition is "all"/empty
            - Provides automatic labeling for common conditions

        Error Handling:
            - Validates data presence and required columns
            - Handles empty filtered subsets
            - Provides specific feedback for:
              * Unparsable conditions/criteria
              * Missing data columns
              * Empty result sets

        Example Outputs:
            1. Successful calculation:
               "25.5% of expert freelancers completed more than 10 projects."
            2. Error cases:
               - "No freelancers found matching the condition 'experts'."
               - "Could not parse criteria 'over 5 projects'..."

        Notes:
            - Always copies data before filtering to preserve original
            - Uses case-insensitive matching for conditions/criteria
            - Logs warnings for unprocessable inputs
        """
        if self.data is None:
            return "Error: Data not loaded for percentage calculation."

        subset = self.data.copy()
        condition_label_for_output = "all"  # Default

        condition_lower = condition.lower().strip()
        condition_filter = self._create_group_filter(condition_lower)

        if condition_filter is not None:
            subset = subset[condition_filter]
            # Determine a cleaner label for output
            if "expert" in condition_lower or "advanced" in condition_lower:
                condition_label_for_output = "expert"
            elif "beginner" in condition_lower:
                condition_label_for_output = "beginner"
            elif (
                condition_lower
            ):  # Use the provided condition if it was specific and resulted in a filter
                condition_label_for_output = condition  # Or a cleaned version of it
        elif condition_lower not in [
            "freelancers",
            "all",
            "",
        ]:
            logger.warning(
                f"Condition '{condition}' not parsed into a filter"
                f" for percentage calculation. "
                f"Applying criteria to '{condition_label_for_output}' freelancers."
            )

        if subset.empty:
            if condition_filter is not None:
                return (
                    f"No freelancers found matching the condition"
                    f"'{
                        condition_label_for_output if condition_label_for_output
                                                      != 'all' else condition}'."
                )
            else:
                return "No data available to calculate percentage."

        criteria_lower = criteria.lower().strip()
        comparison_type = None
        n_projects = None

        # Ensure 'projects_completed' column exists
        if "projects_completed" not in subset.columns:
            logger.error(
                "'projects_completed' column missing, cannot calculate"
                " project-based percentage."
            )
            return "Error: 'projects_completed' column is missing."

        # Criteria parsing (simplified, assuming the criteria structure
        # is "VERB COMPARATOR NUMBER projects")
        m = re.search(
            r"(completed\s+|)(less\s+than|more\s+than|exactly)\s+(\d+)(?:\s+projects)?",
            criteria_lower,
        )
        if m:
            comparison_type = m.group(2).strip()  # "less than", "more than", "exactly"
            n_projects = int(m.group(3))

        if comparison_type and n_projects is not None:
            count = 0
            if comparison_type == "less than":
                count = len(subset[subset["projects_completed"] < n_projects])
            elif comparison_type == "more than":
                count = len(subset[subset["projects_completed"] > n_projects])
            elif comparison_type == "exactly":
                count = len(subset[subset["projects_completed"] == n_projects])

            total_in_subset = len(subset)
            if total_in_subset == 0:
                return (
                    f"No freelancers in the group '{condition_label_for_output}'"
                    f" to calculate percentage from (empty subset after filtering)."
                )

            pct = (count / total_in_subset) * 100 if total_in_subset > 0 else 0
            return (
                f"{pct:.1f}% of {condition_label_for_output} freelancers "
                f"completed {comparison_type} {n_projects} projects."
            )

        # Placeholder for other criteria types (e.g., percentage by payment method)
        # ...

        logger.warning(
            f"Could not parse criteria '{criteria}' for percentage calculation on"
            f" condition '{condition_label_for_output}'."
        )
        return (
            f"Could not calculate the requested percentage. The criteria"
            f" '{criteria}' might not be supported for the condition"
            f" '{condition_label_for_output}'."
        )

    def get_single_group_average_income(self, group_description: str) -> str:
        """Calculates and formats the average income for a specified freelancer group.

        Features:
        - Parses natural language group descriptions
        - Handles all edge cases with clear error messages
        - Provides properly formatted monetary output

        Args:
            group_description: Natural language description of the target group:
                             - "expert freelancers"
                             - "crypto users"
                             - "mobile banking payers"
                             Supports all patterns recognized by _create_group_filter

        Returns:
            str: Formatted result containing either:
                 - Average income with proper USD formatting, or
                 - Specific error message if calculation fails

        Calculation Process:
            1. Parses group description into filter
            2. Validates filter results in non-empty group
            3. Checks income data availability and type
            4. Calculates mean income
            5. Formats result with error handling at each step

        Error Handling:
            - Missing data
            - Unrecognized group descriptions
            - Empty groups
            - Invalid income data
            - NaN results

        Example Outputs:
            1. Success: "The average income for... is $5,234.56."
            2. Errors:
               - "Could not identify the group..."
               - "No freelancers found matching..."
               - "Income data for... is missing or not numeric."

        Notes:
            - Uses mean() for average calculation
            - Returns USD-formatted values with 2 decimal places
            - Handles NaN/None values in income data
            - Case-insensitive group description matching
        """
        if self.data is None:
            return "Error: Data not loaded for average income calculation."

        parsed_group_filter = self._create_group_filter(group_description)

        if parsed_group_filter is None:
            return (
                f"Could not identify the group '{group_description}'"
                f" to calculate average income."
            )

        # Check if filter results in any data BEFORE trying to access columns
        if not self.data[parsed_group_filter].shape[0] > 0:
            return f"No freelancers found matching '{group_description}'."

        # Now safe to assume data exists for the group
        filtered_data = self.data.loc[
            parsed_group_filter
        ]  # Use .loc for explicit indexing

        if "income" not in filtered_data.columns or not pd.api.types.is_numeric_dtype(
            filtered_data["income"]
        ):
            return f"Income data for '{group_description}' is missing or not numeric."

        mean_income = filtered_data["income"].mean()

        if pd.isna(
            mean_income
        ):  # Could happen if all incomes in the filtered group are NaN
            return (
                f"Average income for '{group_description}' could not be calculated"
                f" (e.g., all income data for this group is invalid/NaN)."
            )

        return (
            f"The average income for freelancers described as '{group_description}'"
            f" is ${mean_income:,.2f}."
        )

    def get_most_common(self, column_name: str) -> str:
        """Identifies and returns the most frequent value(s) in a specified column.

        Features:
        - Flexible column name matching (case-insensitive partial matching)
        - Handles multiple modes (ties for most frequent)
        - Comprehensive error handling for data issues

        Args:
            column_name: Name or partial name of column to analyze. Supports:
                       - Exact matches ('payment_method')
                       - Partial matches ('payment' → 'payment_method')
                       - Common aliases mapped in cleaned_column_name_map

        Returns:
            str: Formatted result containing either:
                 - The most common value(s) with frequencies, or
                 - Specific error message if analysis fails

        Supported Columns:
            - job_category
            - payment_method
            - region
            (Additional columns can be added to cleaned_column_name_map)

        Calculation Method:
            1. Maps input to actual DataFrame column name
            2. Uses pandas.Series.mode() which:
               - Returns all values with highest frequency
               - Handles ties automatically
               - Returns empty Series for all-unique columns

        Error Handling:
            - Missing/empty data
            - Unrecognized columns
            - All-unique columns
            - Empty results

        Example Outputs:
            1. Single mode: "The most common payment_method(s): credit_card."
            2. Multiple modes: "The most common region(s): NA, EU."
            3. Errors:
               - "Cannot find the most common value..."
               - "No data available in column..."
               - "Could not determine the most common..."

        Notes:
            - Returns all modes if there are ties
            - Uses string representation of values
            - Empty/NULL values are excluded from mode calculation
            - Column matching is case-insensitive
        """
        if self.data is None:
            return "Error: Data not loaded."

        cleaned_column_name_map = {
            "job_category": "job_category",
            "payment_method": "payment_method",
            "region": "region",
        }
        actual_col_name_in_df = None
        for key, val in cleaned_column_name_map.items():
            if key in column_name.lower():  # `column_name` from parser
                actual_col_name_in_df = val
                break

        if not actual_col_name_in_df or actual_col_name_in_df not in self.data.columns:
            return (
                f"Cannot find the most common value for '{column_name}'"
                f" - column not recognized or not available."
            )

        if self.data[actual_col_name_in_df].empty:
            return f"No data available in column '{actual_col_name_in_df}'."

        mode_series = self.data[actual_col_name_in_df].mode()
        if not mode_series.empty:
            # .mode() can return multiple values if they have the same highest frequency
            modes = ", ".join(mode_series.astype(str).tolist())
            return f"The most common {column_name}(s): {modes}."
        else:
            return (
                f"Could not determine the most common {column_name}"
                f" (column might be all unique or empty)."
            )

    def get_unique_values(self, column_name: str) -> str:
        """Retrieves and formats the unique values from a specified dataset column.

        Features:
        - Handles both categorical and numerical columns
        - Special formatting for 'expert_status' with descriptive mapping
        - Automatic sorting and display limiting
        - Comprehensive error handling

        Args:
            column_name: Name of the column to analyze (case-sensitive)
                       Must exactly match a DataFrame column name

        Returns:
            str: Formatted result containing either:
                 - Sorted list of unique values (limited to 20 items)
                 - Descriptive labels for special columns
                 - Specific error message if analysis fails

        Special Cases:
            - 'expert_status': Converts binary values to descriptive labels
                              (0 → Beginner/Intermediate, 1 → Expert/Advanced)

        Output Formatting:
            - Sorts values alphabetically/numerically
            - Converts all values to strings
            - Excludes NaN/None values
            - Limits display to 20 items with overflow indicator
            - Human-friendly column names in output

        Error Handling:
            - Missing/empty data
            - Unrecognized columns
            - All-NaN columns
            - Empty results

        Example Outputs:
            1. Regular column: "Available Payment Methods:
            credit_card, paypal, wire_transfer"
            2. Expert status: "Available Experience Levels:
            Beginner/Intermediate, Expert/Advanced"
            3. Many values: "Available Job Categories: design, dev, ... and 15 more..."
            4. Errors:
               - "Error: Column 'skills' not found..."
               - "No data available in column..."
               - "All values in 'region' are missing..."

        Notes:
            - Uses exact column name matching (case-sensitive)
            - Maintains consistency with _clean_data's expert_status interpretation
            - Returns empty string representation for NaN/None values
        """
        if self.data is None:
            return "Error: Data not loaded."
        if column_name not in self.data.columns:
            return f"Error: Column '{column_name}' not found in the dataset."

        if self.data[column_name].empty:
            return f"No data available in column '{column_name}' to list unique values."

        unique_values = self.data[column_name].unique()

        # Special handling for 'expert_status' if you want more descriptive output
        if column_name == "expert_status":
            # Assuming 0 is Beginner, 1 is Expert/Advanced based on your _clean_data
            # This mapping should be consistent with how you interpret
            # the binary values.
            status_map = {0: "Beginner/Intermediate", 1: "Expert/Advanced"}
            descriptive_values = sorted(
                [status_map.get(val, str(val)) for val in unique_values]
            )
            return f"Available experience levels: {', '.join(descriptive_values)}."

        if len(unique_values) > 0:
            # Sort for consistent output, convert to string
            sorted_unique_values = sorted(
                [str(val) for val in unique_values if pd.notna(val)]
            )
            if not sorted_unique_values:  # All values were NaN
                return f"All values in '{column_name}' are missing or undefined."

            # Limit the number of displayed items if too many
            max_display = 20
            if len(sorted_unique_values) > max_display:
                displayed_values = (
                    ", ".join(sorted_unique_values[:max_display])
                    + f", and {len(sorted_unique_values) - max_display} more..."
                )
            else:
                displayed_values = ", ".join(sorted_unique_values)
            return (
                f"Available {column_name.replace('_', ' ').title()}s:"
                f" {displayed_values}."
            )
        else:
            return (
                f"No unique values found in column '{column_name}'"
                f" (it might be empty or all NaN)."
            )
