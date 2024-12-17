# test_speaker_identification.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_speaker_profiles(diarizer: SpeakerDiarizer, reference_dir: Path):
    """Add speaker profiles from reference audio files."""
    logger.info("Adding speaker profiles...")
    
    # Assuming reference audio files are named like "speaker_name_sample1.mp3"
    for audio_file in reference_dir.glob("*.mp3"):
        try:
            # Extract speaker name from filename
            speaker_name = audio_file.stem.split("_")[0]
            logger.info(f"Adding profile for {speaker_name} from {audio_file}")
            diarizer.add_speaker_profile(speaker_name, audio_file)
            logger.info(f"Successfully added profile for {speaker_name}")
        except Exception as e:
            logger.error(f"Failed to add profile from {audio_file}: {str(e)}")

def main():
    # Your HuggingFace token
    auth_token = "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"
    
    # Initialize diarizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarizer = SpeakerDiarizer(auth_token=auth_token, device=device)
    
    # Setup directories
    reference_dir = Path("data/reference_audio")
    reference_dir.mkdir(parents=True, exist_ok=True)
    profiles_file = Path("data/speaker_profiles.pkl")
    test_audio = Path("data/audio/test.mp3")
    
    # Check if we have saved profiles
    if profiles_file.exists():
        logger.info("Loading existing speaker profiles...")
        diarizer.load_speaker_profiles(profiles_file)
    else:
        # Add new speaker profiles
        add_speaker_profiles(diarizer, reference_dir)
        logger.info("Saving speaker profiles...")
        diarizer.speaker_identifier.save_profiles(profiles_file)
    
    # Test identification
    logger.info(f"Processing test audio: {test_audio}")
    segments = diarizer.diarize(test_audio)
    
    # Print results
    print("\nDiarization Results:")
    print("=" * 80)
    for segment in segments:
        confidence = getattr(segment, 'confidence', 'N/A')
        print(f"[{segment.start:.1f}s - {segment.end:.1f}s] "
              f"Speaker: {segment.speaker} (Confidence: {confidence})")
        if hasattr(segment, 'text') and segment.text:
            print(f"Text: {segment.text}")
        print("-" * 80)
    
    # Print statistics
    unique_speakers = set(segment.speaker for segment in segments)
    print(f"\nDetected {len(unique_speakers)} unique speakers:")
    for speaker in sorted(unique_speakers):
        speaker_segments = [s for s in segments if s.speaker == speaker]
        total_time = sum(s.end - s.start for s in speaker_segments)
        print(f"- {speaker}: {total_time:.1f} seconds of speech")

if __name__ == "__main__":
    main()