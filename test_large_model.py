# test_large_model.py
from src.audio.transcriber import WhisperTranscriber
from pathlib import Path
import time

def main():
    # Initialize transcriber with large model
    print("Initializing transcriber with large model...")
    transcriber = WhisperTranscriber(model_name="large")
    
    # Path to your audio file
    audio_path = Path("data/audio/test.mp3")
    
    print(f"\nAttempting to transcribe: {audio_path}")
    
    # Time the transcription
    start_time = time.time()
    
    try:
        # First, try with preprocessing
        print("\nTranscribing with preprocessing...")
        result = transcriber.transcribe(
            audio_path,
            language="en",
            preprocess=True
        )
        
        print("\nTranscription with preprocessing:")
        print("-" * 80)
        print(result["text"])
        print("-" * 80)
        
        # Calculate and print duration
        duration = time.time() - start_time
        print(f"\nTranscription took {duration:.2f} seconds")
        
        # Print some segments with timestamps
        print("\nSample segments with timestamps:")
        segments = transcriber.get_segments(result)
        for segment in segments[:5]:
            print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()