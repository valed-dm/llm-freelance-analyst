import importlib.util  # For checking safetensors availability
import logging

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import pipeline

from src.core.decorators import log_execution


logger = logging.getLogger(__name__)


class LLMTranslator:
    """
    Handles translation of a text from a source language to a target language
    using specified Hugging Face language models.
    """

    # Default models, can be expanded
    DEFAULT_MODEL_MAPPING = {
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
        # "en-es": "Helsinki-NLP/opus-mt-en-es", # Example for another pair
    }

    def __init__(self, model_mapping: dict = None, device: str = None):
        """
        Initializes the LLMTranslator with specified translation models.

        Args:
            model_mapping (dict, optional): A dictionary where keys are "source_lang-target_lang"
                                            (e.g., "ru-en") and values are Hugging Face model
                                            identifiers. Defaults to LLMTranslator.DEFAULT_MODEL_MAPPING.
            device (str, optional): The device to run the models on (e.g., "cpu", "cuda", "mps").
                                    If None, transformers pipeline will attempt auto-detection.
        """  # noqa: E501
        self.model_mapping = (
            model_mapping if model_mapping else self.DEFAULT_MODEL_MAPPING
        )
        self.device_to_use = device
        # Stores loaded pipelines, e.g., {"ru-en": pipeline_object}
        self.translation_pipelines = {}

        self._load_models()

    def _load_models(self):
        """
        Loads the translation models and tokenizers for each language pair
        defined in self.model_mapping and initializes their pipelines.
        """
        _safetensors_available = importlib.util.find_spec("safetensors") is not None
        if not _safetensors_available:
            logger.warning(
                "LLMTranslator: 'safetensors' library not found. "
                "Model loading might fall back to .bin files,"
                " which could trigger PyTorch version errors. "
                "Consider installing with: pip install safetensors"
            )

        for lang_pair, model_name in self.model_mapping.items():
            try:
                logger.info(
                    f"LLMTranslator: Attempting to load model '{model_name}'"
                    f" for '{lang_pair}' "
                    f"with safetensors preference (available: {_safetensors_available}"
                    f")."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_obj = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, use_safetensors=_safetensors_available
                )

                pipeline_device_arg = None
                if self.device_to_use:
                    if self.device_to_use.startswith("cuda") or self.device_to_use in [
                        "cpu",
                        "mps",
                    ]:
                        pipeline_device_arg = self.device_to_use
                    else:
                        try:
                            pipeline_device_arg = int(self.device_to_use)
                        except ValueError:
                            logger.warning(
                                f"LLMTranslator: Invalid device '{self.device_to_use}'"
                                f" for '{model_name}'. Letting pipeline auto-detect."
                            )

                # For Helsinki-NLP models, simply using "translation"
                # as the task is robust.
                # The model itself knows its source and target languages.
                # If you needed to be more specific for other model types,
                # you could construct the task name:
                # task_name = f"translation_{lang_pair.replace('-', '_to_')}"
                # if len(lang_pair.split('-')) == 2 else "translation"
                # For now, we'll stick with "translation" as it's generally reliable
                # for Helsinki models.
                pipeline_task = "translation"

                loaded_pipeline = pipeline(
                    pipeline_task,
                    model=model_obj,
                    tokenizer=tokenizer,
                    device=pipeline_device_arg,
                )
                self.translation_pipelines[lang_pair] = loaded_pipeline
                logger.info(
                    f"LLMTranslator: Model '{model_name}' for '{lang_pair}'"
                    f" loaded successfully "
                    f"on device '{loaded_pipeline.device}' using task"
                    f" '{pipeline_task}'."
                )
            except Exception as e:
                logger.exception(
                    f"LLMTranslator: Error loading translation model '{model_name}'"
                    f" for '{lang_pair}': {e}.\n"
                    f"Ensure the model is valid. Translation for '{lang_pair}'"
                    f" will be unavailable."
                )
                self.translation_pipelines[lang_pair] = None

    @log_execution()
    def translate(
        self, query_text: str, source_lang: str, target_lang: str
    ) -> str | None:
        """
        Translates the given query text from source_lang to target_lang.

        Args:
            query_text (str): The text to translate.
            source_lang (str): The source language code (e.g., "ru").
            target_lang (str): The target language code (e.g., "en").

        Returns:
            str | None: The translated text, or None if translation fails or the
                        required language pair model is not loaded.
                        Returns original query_text if source and target are the same.
        """
        if not query_text:
            logger.warning("LLMTranslator: Empty query text received for translation.")
            return query_text  # Return empty string as is, or None

        if source_lang == target_lang:
            logger.debug(
                f"LLMTranslator: Source and target languages are the same"
                f" ('{source_lang}'). No translation needed."
            )
            return query_text

        lang_pair_key = f"{source_lang}-{target_lang}"
        selected_pipeline = self.translation_pipelines.get(lang_pair_key)

        if selected_pipeline is None:
            logger.error(
                f"LLMTranslator: No translation model loaded for '{lang_pair_key}'. "
                f"Cannot translate query: '{query_text}'"
            )
            # Fallback: Return original query if no translator for the pair
            # This allows the app to proceed with the original query, which might then
            # be handled by logic that expects the target_lang.
            # Or, you could raise an error or return None strictly.
            logger.warning(
                f"LLMTranslator: Returning original query due to missing translator"
                f" for '{lang_pair_key}'."
            )
            return query_text

        try:
            logger.debug(
                f"LLMTranslator: Translating from '{source_lang}' to '{target_lang}':"
                f" '{query_text}'"
            )
            # Helsinki-NLP models often don't need src_lang/tgt_lang specified in call
            # if the model is pair-specific
            # The pipeline should infer it. If not, some models might take `src_lang`
            # and `tgt_lang` args.
            # For `pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")`,
            # it knows its job.
            translated_output = selected_pipeline(query_text)

            if (
                translated_output
                and isinstance(translated_output, list)
                and len(translated_output) > 0
            ):
                translation = translated_output[0].get("translation_text")
                if translation:
                    logger.debug(f"LLMTranslator: Translated text: '{translation}'")
                    return translation
                else:
                    logger.error(
                        "LLMTranslator: 'translation_text' not found in model output."
                    )
                    return None  # Or original text as fallback
            else:
                logger.error(
                    f"LLMTranslator: Unexpected translation model output format:"
                    f" {translated_output}"
                )
                return None  # Or original text
        except Exception as e:
            logger.exception(
                f"LLMTranslator: Error during translation from"
                f" '{source_lang}' to '{target_lang}' "
                f"for query '{query_text}': {e}"
            )
            return None  # Or original text


# Example usage for testing this class in isolation:
if __name__ == "__main__":
    try:
        # Metal Performance Shaders (MPS) backend for GPU training acceleration on Mac:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        translator = LLMTranslator(device=device)

        # translator = LLMTranslator() # Uses default ru-en model, auto device

        if translator.translation_pipelines.get("ru-en"):  # Check if ru-en loaded
            test_query_ru = "Привет, мир!"
            translated_text = translator.translate(
                test_query_ru,
                source_lang="ru",
                target_lang="en",
            )
            print(f"Original (ru): '{test_query_ru}'")
            print(f"Translated (en): '{translated_text}'")

            # Test non-translation case
            test_query_en = "Hello, world!"
            translated_text_en = translator.translate(
                test_query_en,
                source_lang="en",
                target_lang="en",
            )
            print(f"Original (en): '{test_query_en}'")
            print(f"Translated (en): '{translated_text_en}'")

            # Test unsupported pair (will return original text due to current fallback)
            translated_text_unsupported = translator.translate(
                test_query_ru,
                source_lang="ru",
                target_lang="es",
            )
            print(f"Original (ru) for es: '{test_query_ru}'")
            # Should be original
            print(f"Translated (es): '{translated_text_unsupported}'")
        else:
            print("Translator pipeline for 'ru-en' failed to load. Cannot run tests.")

    except Exception as main_e:
        print(f"Error in example usage: {main_e}")
