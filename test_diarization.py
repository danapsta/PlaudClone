# test_diarization.py
from src.audio.processor import AudioProcessor
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Your HuggingFace token
    auth_token = "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"
    
    # Initialize processor
    print("Initializing audio processor...")
    processor = AudioProcessor(auth_token=auth_token)
    
    # Process audio file
    audio_path = Path("data/audio/test.mp3")
    
    print(f"\nProcessing audio file: {audio_path}")
    result = processor.process_audio(audio_path, language="en")
    
    # Print results
    print("\nFormatted transcript with speaker labels:")
    print("=" * 80)
    print(result["formatted_transcript"])
    print("=" * 80)
    
    # Print speaker statistics
    speakers = set(segment.speaker for segment in result["speaker_segments"])
    print(f"\nDetected {len(speakers)} speakers in the audio")
    
if __name__ == "__main__":
    main()