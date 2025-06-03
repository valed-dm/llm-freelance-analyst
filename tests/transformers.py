import time

from huggingface_hub import snapshot_download  # For a more robust manual download
import requests
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline


MODEL_NAME = "gpt2"  # Or "gpt2-medium", "gpt2-large", etc.
FORCE_DOWNLOAD = True  # Set to True to ensure files are re-downloaded

TIMEOUT = 300  # 5-minute timeout for requests


def verify_connectivity():
    """Check if we can actually reach Hugging Face API"""
    try:
        print("\n🔍 Testing Hugging Face API connectivity...")
        # Test a general API endpoint or the specific model's API
        test_url = f"https://huggingface.co/api/models/{MODEL_NAME}"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Can access model metadata for {MODEL_NAME}")
            return True
        print(f"❌ API for {MODEL_NAME} returned {response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"🚨 Connectivity test failed: {str(e)}")
        return False


def download_model_manually_to_path(model_name, save_path):
    """
    Downloads the entire model snapshot to a specified path using huggingface_hub.
    This ensures all necessary files (config, tokenizer, weights) are present.
    """
    print(f"\n📥 Manually downloading {model_name} to {save_path}...")
    try:
        snapshot_location = snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ Model {model_name} downloaded to {snapshot_location}")
        return snapshot_location
    except Exception as e:
        print(f"🚨 Manual download of {model_name} failed: {str(e)}")
        return None


def init_generator():
    if not verify_connectivity():
        return None

    print("\n⏳ Starting model initialization (may take several minutes)...")
    if FORCE_DOWNLOAD:
        print(
            "⚡ FORCE_DOWNLOAD is True. Model files will be re-downloaded,"
            " ignoring cache."
        )

    model_load_path = MODEL_NAME
    tokenizer_load_path = MODEL_NAME

    try:
        with tqdm(total=4, desc="Initialization Progress") as main_bar:
            main_bar.set_description(f"Configuring model: {model_load_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map="auto",  # Automatically uses GPU if available via accelerate
                force_download=FORCE_DOWNLOAD,
            )
            main_bar.update(1)

            main_bar.set_description(f"Loading tokenizer: {tokenizer_load_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_load_path,
                force_download=FORCE_DOWNLOAD,
            )
            main_bar.update(1)

            main_bar.set_description("Building pipeline")
            # When the model is loaded with device_map="auto",
            # do NOT pass the 'device' argument to the pipeline.
            # It will infer the device(s) from the model.
            generator = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, framework="pt"
            )
            main_bar.update(1)

            main_bar.set_description("Running test inference")
            _ = generator("Test", max_new_tokens=1)
            main_bar.update(1)

            return generator

    except Exception as e:
        print(f"\n🔥 Initialization failed: {str(e)}")
        print("\nTry these solutions:")
        print("1. Check your internet connection.")
        print(
            "2. Ensure you have enough disk space in your Hugging Face cache"
            " (usually ~/.cache/huggingface/hub)."
        )
        print(
            f"3. If issues persist, try deleting the cache for {MODEL_NAME}"
            f" and run again."
        )
        print(
            "   Cache location is typically "
            "~/.cache/huggingface/hub/models--<model_name_with_double_hyphen_for_slash>"
        )
        print("   Example for 'gpt2': ~/.cache/huggingface/hub/models--gpt2")
        print("4. Run 'pip install -U transformers huggingface_hub torch'")
        print("5. Temporarily disable VPN/Firewall if they might be interfering.")
        print(
            f"6. Manually download from Hugging Face website: https://huggingface.co/{MODEL_NAME}/tree/main"
        )
        return None


if __name__ == "__main__":
    print(f"🚀 Starting GPT Load Test for: {MODEL_NAME}")
    if FORCE_DOWNLOAD:
        print("⚠️ FORCE_DOWNLOAD is enabled. Cache will be ignored for this run.")

    # Simplified system check
    print("\n💓 System check (cosmetic):")
    for i in range(1, 2):  # just to show it starts
        time.sleep(0.1)
        print(f"• Step {i}/1")

    generator = init_generator()

    if generator:
        print("\n🎉 System ready! Test prompt:")
        try:
            # Simple generation without tqdm callback for clarity
            prompt = "Hello, AI! Please respond with 'Operational'"
            print(f'   Prompt: "{prompt}"')
            response = generator(prompt, max_new_tokens=20)
            print("\n   Response: " + response[0]["generated_text"])
        except Exception as e:
            print(f"\n⚠️ Generation failed: {str(e)}")
    else:
        print("\n❌ Initialization failed - see above errors for details.")
