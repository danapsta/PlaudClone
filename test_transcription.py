# test_transcription.py
from src.audio.transcriber import WhisperTranscriber
from pathlib import Path
import os

def main():
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Try different path variations
    relative_path = Path("data/audio/test.mp3")
    absolute_path = Path("C:/Users/bayat/Desktop/Code/PlaudClone/data/audio/test.mp3")
    project_root_path = Path(__file__).parent / "data" / "audio" / "test.mp3"
    
    print("\nTesting different paths:")
    print(f"Relative path: {relative_path}")
    print(f"Absolute path: {absolute_path}")
    print(f"Project root path: {project_root_path}")
    
    print("\nChecking if paths exist:")
    print(f"Relative path exists: {relative_path.exists()}")
    print(f"Absolute path exists: {absolute_path.exists()}")
    print(f"Project root path exists: {project_root_path.exists()}")
    
    # Initialize transcriber
    print("\nInitializing transcriber...")
    transcriber = WhisperTranscriber(model_name="medium")
    
    # Try to transcribe using absolute path
    try:
        print(f"\nAttempting transcription with absolute path...")
        result = transcriber.transcribe(absolute_path)
        
        # Print full transcription
        print("\nFull Transcription:")
        print("-" * 80)
        print(result["text"])
        print("-" * 80)
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()