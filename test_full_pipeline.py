# test_full_pipeline.py
from src.audio.processor import AudioProcessor  # Using the combined processor
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_pipeline():
    auth_token = "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"  # Replace with your token
    
    try:
        # Initialize processor
        processor = AudioProcessor(
            auth_token=auth_token,
            whisper_model="large",  # Using the large model for better accuracy
            device="cuda"
        )
        
        # Process test audio
        audio_path = Path("data/audio/test.mp3")
        logger.info(f"\nProcessing file: {audio_path}")
        
        # Time the processing
        start_time = time.time()
        
        # Process audio (this will do both diarization and transcription)
        result = processor.process_audio(audio_path, language="en")
        
        # Print results
        duration = time.time() - start_time
        print(f"\nProcessing completed in {duration:.2f} seconds")
        
        segments = result["speaker_segments"]
        print(f"Found {len(segments)} segments")
        
        # Print first few segments with transcriptions
        print("\nFirst few segments with transcriptions:")
        print("=" * 80)
        for segment in segments[:10]:  # Show first 10 segments
            print(f"[{segment.start:.1f}s - {segment.end:.1f}s] "
                  f"Speaker: {segment.speaker}")
            if segment.text:
                print(f"Text: {segment.text}")
            print("-" * 80)
        
        # Print speaker statistics
        unique_speakers = set(s.speaker for s in segments)
        print(f"\nDetected {len(unique_speakers)} unique speakers:")
        for speaker in sorted(unique_speakers):
            speaker_segments = [s for s in segments if s.speaker == speaker]
            total_time = sum(s.end - s.start for s in speaker_segments)
            print(f"\n{speaker}:")
            print(f"- Speaking time: {total_time:.1f} seconds")
            print(f"- Number of turns: {len(speaker_segments)}")
            # Print a sample of what this speaker said
            if speaker_segments[0].text:
                print(f"- Sample: \"{speaker_segments[0].text}\"")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_full_pipeline()