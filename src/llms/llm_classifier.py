import importlib.util  # For checking safetensors availability
import logging

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline

from src.core.decorators import log_execution


logger = logging.getLogger(__name__)


class LLMClassifier:
    """
    Handles the classification of user queries into predefined categories
    using a specified language model.
    """

    DEFAULT_MODEL_NAME = "google/flan-t5-small"
    DEFAULT_PROMPT_TEMPLATE = """Classify this freelancer data query into one of these types.
            If the query asks for a direct comparison of income between two groups, use 'income_comparison'.
            If the query asks for how income is spread across categories, use 'income_distribution'.
            If the query explicitly asks for a 'percentage' of a group meeting a condition, use 'percentage_calculation'.
            If the query asks to find the most frequent or common item in a category, use 'mode_calculation'.
            If the query asks for an 'average income' or 'mean income' of a single group, use 'average_income_calculation'.
            For all other types of queries, especially general information requests not covered by the above, use 'other'.

            - income_comparison (e.g., 'Compare income of X versus Y')
            - income_distribution (e.g., 'Show income distribution by region')
            - percentage_calculation (e.g., 'What percentage of experts did Z?')
            - average_income_calculation (e.g., 'What is the average income of group X?', 'Find the mean earnings for experts.')
            - mode_calculation (e.g., 'What is the most common job category?', 'Most frequent payment method')
            - list_unique_values (e.g., 'What are all job categories?', 'List all payment methods', 'Show distinct client regions.')
            - other (e.g., 'Tell me about freelancers')

            Query: {query}"""  # noqa: E501

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initializes the LLMClassifier with a specified model.

        Args:
            model_name (str, optional):
                The Hugging Face model identifier for classification.
                Defaults to LLMClassifier.DEFAULT_MODEL_NAME.
            device (str, optional):
                The device to run the model on (e.g., "cpu", "cuda", "mps").
                If None, transformers pipeline will attempt auto-detection.
        """
        self.model_name = model_name if model_name else self.DEFAULT_MODEL_NAME
        self.device_to_use = device  # Can be None for auto-detection by pipeline

        self.classification_pipeline = None
        self._load_model()

    def _load_model(self):
        """
        Loads the classification model and tokenizer and initializes the pipeline.
        Prioritizes using safetensors if available.
        """
        _safetensors_available = importlib.util.find_spec("safetensors") is not None
        if not _safetensors_available:
            logger.warning(
                "LLMClassifier: 'safetensors' library not found. "
                "Model loading might fall back to .bin files,"
                " which could trigger PyTorch version errors. "
                "Consider installing with: pip install safetensors"
            )

        try:
            logger.info(
                f"LLMClassifier: Attempting to load model '{self.model_name}' "
                f"with safetensors preference (available: {_safetensors_available})."
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, use_safetensors=_safetensors_available
            )

            pipeline_device_arg = None
            if self.device_to_use:
                if self.device_to_use.startswith("cuda") or self.device_to_use in [
                    "cpu",
                    "mps",
                ]:
                    pipeline_device_arg = self.device_to_use
                else:  # Assuming it might be an integer for GPU
                    try:
                        pipeline_device_arg = int(self.device_to_use)
                    except ValueError:
                        logger.warning(
                            f"Invalid device specified: {self.device_to_use}."
                            f" Letting pipeline auto-detect."
                        )

            self.classification_pipeline = pipeline(
                "text2text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                device=pipeline_device_arg,  # Pass device to pipeline
            )
            logger.info(
                f"LLMClassifier: Model '{self.model_name}' loaded successfully "
                f"on device '{self.classification_pipeline.device}'."
            )
        except Exception as e:
            logger.exception(
                f"LLMClassifier: Error loading classification model"
                f" '{self.model_name}': {e}.\n"
                f"Ensure the model is valid and supports safetensors if applicable, "
                f"or that PyTorch version meets requirements for .bin files."
            )
            raise

    @log_execution()
    def classify(
        self,
        query_text: str,
        prompt_template: str = None,
        max_new_tokens: int = 20,
    ) -> str | None:
        """
        Classifies the given query text using the loaded LLM.

        Args:
            query_text (str): The user's query to classify.
            prompt_template (str, optional): A custom prompt template string.
                                             Must contain a '{query}' placeholder.
                                             Defaults to LLMClassifier.DEFAULT_PROMPT_TEMPLATE.
            max_new_tokens (int, optional): Max new tokens for the LLM to generate for classification.
                                            Defaults to 20.

        Returns:
            str | None: The raw classification string from the LLM (e.g., "income_comparison"),
                        or None if classification fails or pipeline not loaded.
        """  # noqa: E501
        if self.classification_pipeline is None:
            logger.error(
                "LLMClassifier: Classification pipeline not loaded."
                " Cannot classify query."
            )
            return None
        if not query_text:
            logger.warning(
                "LLMClassifier: Empty query text received for classification."
            )
            return "other"  # Or None, or a specific "empty_query" classification

        current_prompt_template = (
            prompt_template if prompt_template else self.DEFAULT_PROMPT_TEMPLATE
        )

        try:
            prompt = current_prompt_template.format(query=query_text)
        except KeyError:
            logger.error(
                f"LLMClassifier: Prompt template requires a '{{query}}' placeholder. "
                f"Used template: '{current_prompt_template}'"
            )
            return None  # Or raise an error

        try:
            logger.debug(f"LLMClassifier: Sending prompt to model:\n{prompt}")
            # The pipeline returns a list of dictionaries
            raw_outputs = self.classification_pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
            )

            if raw_outputs and isinstance(raw_outputs, list) and len(raw_outputs) > 0:
                generated_text = raw_outputs[0].get("generated_text")
                if generated_text:
                    logger.debug(f"LLMClassifier: Raw model output: '{generated_text}'")
                    return generated_text.strip().lower()  # Return cleaned output
                else:
                    logger.error(
                        "LLMClassifier: 'generated_text' not found in model output."
                    )
                    return None
            else:
                logger.error(
                    f"LLMClassifier: Unexpected model output format: {raw_outputs}"
                )
                return None
        except Exception as e:
            logger.exception(
                f"LLMClassifier: Error during query classification for query"
                f" '{query_text}': {e}"
            )
            return None


# Example usage for testing this class in isolation:
if __name__ == "__main__":
    try:
        # Metal Performance Shaders (MPS) backend for GPU training acceleration on Mac:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        classifier = LLMClassifier(device=device)

        # classifier = LLMClassifier()  # Uses default model and auto device

        if classifier.classification_pipeline:  # Check if loaded
            test_query = "What is the average income of experts?"
            classification = classifier.classify(test_query)
            print(f"Query: '{test_query}'")
            print(f"Classification: '{classification}'")

            test_query_2 = "Compare income of paypal users vs crypto users"
            classification_2 = classifier.classify(test_query_2)
            print(f"Query: '{test_query_2}'")
            print(f"Classification: '{classification_2}'")
        else:
            print("Classifier pipeline failed to load. Cannot run tests.")

    except Exception as main_e:
        print(f"Error in example usage: {main_e}")
