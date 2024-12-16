# test_pipeline.py
from src.audio.processor import AudioProcessor
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)

def main():
    # Your HuggingFace token
    auth_token = "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"
    
    print("Initializing audio processor...")
    try:
        processor = AudioProcessor(
            auth_token=auth_token,
            whisper_model="large",
            device="cuda"
        )
        
        # Process audio file
        audio_path = Path("data/audio/test.mp3")
        
        print(f"\nProcessing audio file: {audio_path}")
        print("This may take a few minutes...\n")
        
        start_time = time.time()
        result = processor.process_audio(audio_path, language="en")
        
        processing_time = time.time() - start_time
        
        # Print results
        print("\nFormatted transcript with speaker labels:")
        print("=" * 80)
        print(result["formatted_transcript"])
        print("=" * 80)
        
        # Print statistics
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        speakers = set(segment.speaker for segment in result["speaker_segments"])
        print(f"Detected {len(speakers)} speakers in the audio")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()