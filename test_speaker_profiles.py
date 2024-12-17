# test_speaker_profiles.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def test_speaker_profiles():
    auth_token = "YOUR_TOKEN_HERE"  # Replace with your token
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(auth_token=auth_token)
    
    # Load profiles
    profiles_path = Path("data/speaker_profiles.pkl")
    if profiles_path.exists():
        diarizer.load_speaker_profiles(profiles_path)
        print("\nLoaded speaker profiles:")
        for name, profile in diarizer.speaker_identifier.speakers.items():
            print(f"- {name} ({len(profile.embeddings)} samples)")
    else:
        print("No speaker profiles found!")
        return
    
    # Test with an audio file
    test_file = Path("data/audio/Ben_Sample1.mp3")
    print(f"\nTesting with audio file: {test_file}")
    
    try:
        segments = diarizer.diarize(test_file)
        print("\nDiarization results:")
        for segment in segments[:5]:  # Show first 5 segments
            print(f"[{segment.start:.1f}s - {segment.end:.1f}s] "
                  f"Speaker: {segment.speaker}")
    except Exception as e:
        print(f"Error during diarization: {e}")

if __name__ == "__main__":
    test_speaker_profiles()