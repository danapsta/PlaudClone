# test_auth_full.py
from huggingface_hub import HfApi, model_info
import logging
from pyannote.audio import Pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_access(api, model_id, token):
    try:
        info = model_info(model_id, token=token)
        print(f"✓ Have access to {model_id}")
        return True
    except Exception as e:
        print(f"✗ No access to {model_id}: {str(e)}")
        return False

def test_auth(token):
    print("Testing Hugging Face authentication and model access...")
    
    # Initialize API
    api = HfApi()
    
    # Verify authentication
    try:
        user = api.whoami(token)
        print(f"Authenticated as: {user['name']}\n")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return False
    
    # Check access to all required models
    required_models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "pyannote/embedding"
    ]
    
    all_access = True
    print("Checking access to required models:")
    for model in required_models:
        if not check_model_access(api, model, token):
            all_access = False
    
    if not all_access:
        print("\n⚠️ Missing access to one or more required models")
        print("Please visit the following URLs and accept the license agreements:")
        print("- https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("- https://huggingface.co/pyannote/segmentation-3.0")
        print("- https://huggingface.co/pyannote/embedding")
        return False
    
    return True

if __name__ == "__main__":
    auth_token = "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"  # Your token
    test_auth(auth_token)