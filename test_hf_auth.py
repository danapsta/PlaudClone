# test_hf_auth.py
from huggingface_hub import HfApi, HfFolder
import sys

def test_hf_connection(token):
    try:
        # Set token
        HfFolder.save_token(token)
        
        # Initialize API
        api = HfApi()
        
        # Test API connection
        print("Testing API connection...")
        user = api.whoami(token)
        print(f"Successfully authenticated as: {user['name']}")
        
        # Check model access
        print("\nChecking model access...")
        print("Attempting to access pyannote/speaker-diarization...")
        model_info = api.model_info("pyannote/speaker-diarization")
        print(f"Model access successful! Model is {'private' if model_info.private else 'public'}")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_hf_auth.py YOUR_TOKEN")
        sys.exit(1)
        
    token = sys.argv[1]
    test_hf_connection(token)