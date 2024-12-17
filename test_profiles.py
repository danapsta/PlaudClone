# test_profiles.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_profiles():
    auth_token = "YOUR_TOKEN_HERE"  # Replace with your token
    profiles_path = Path("data/speaker_profiles.pkl")
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(auth_token=auth_token)
    
    # Check if profiles exist
    if profiles_path.exists():
        diarizer.load_speaker_profiles(profiles_path)
        print("\nLoaded speaker profiles:")
        for name, profile in diarizer.speaker_identifier.speakers.items():
            print(f"- {name}: {len(profile.embeddings)} samples")
    else:
        print("\nNo speaker profiles found!")
        print(f"Expected profile file at: {profiles_path.absolute()}")
    
    return diarizer

if __name__ == "__main__":
    check_profiles()