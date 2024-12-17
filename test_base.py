# test_base.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def test_audio_loading():
    auth_token = "Yhf_rbRuwCCgUJjNlkBqWmjJPvfrEiWrBGXFvz"  # Your HF token
    diarizer = SpeakerDiarizer(auth_token=auth_token)
    
    # Test audio file
    audio_path = Path("data/audio/test.mp3")
    print(f"\nTesting with audio file: {audio_path}")
    print(f"File exists: {audio_path.exists()}")
    
    if audio_path.exists():
        try:
            # Try to convert to WAV
            wav_path = diarizer._convert_to_wav(audio_path)
            print(f"Successfully converted to WAV: {wav_path}")
        except Exception as e:
            print(f"Error converting audio: {e}")

def test_pipeline_loading():
    auth_token = "Yhf_rbRuwCCgUJjNlkBqWmjJPvfrEiWrBGXFvz"  # Your HF token
    
    try:
        diarizer = SpeakerDiarizer(auth_token=auth_token)
        print("Successfully loaded diarization pipeline")
    except Exception as e:
        print(f"Error loading pipeline: {e}")

if __name__ == "__main__":
    test_audio_loading()
    test_pipeline_loading()