# test_audio.py
from src.audio.transcriber import WhisperTranscriber
from pathlib import Path
import os
import soundfile as sf

def check_audio_file(file_path):
    """Verify audio file is readable"""
    try:
        data, samplerate = sf.read(file_path)
        print(f"Successfully read audio file: {file_path}")
        print(f"Sample rate: {samplerate}")
        print(f"Duration: {len(data) / samplerate:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return False

def main():
    # Get absolute path to audio file
    audio_path = Path("data/audio/test.mp3").resolve()
    
    print(f"Audio file path: {audio_path}")
    print(f"File exists: {audio_path.exists()}")
    if audio_path.exists():
        print(f"File size: {audio_path.stat().st_size} bytes")
    
    # Check if file is readable
    print("\nChecking if audio file is readable...")
    if not check_audio_file(audio_path):
        print("Failed to read audio file!")
        return
    
    # Initialize transcriber
    print("\nInitializing transcriber...")
    transcriber = WhisperTranscriber(model_name="medium")
    
    # Attempt transcription
    try:
        print("\nStarting transcription...")
        result = transcriber.transcribe(audio_path)
        
        print("\nTranscription result:")
        print("-" * 80)
        print(result["text"])
        print("-" * 80)
        
    except Exception as e:
        print(f"Transcription error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()