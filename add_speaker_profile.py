# add_speaker_profile.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(audio_path: Path) -> Path:
    """Convert audio file to WAV format if needed."""
    if audio_path.suffix.lower() == '.wav':
        return audio_path
        
    output_path = audio_path.parent / f"{audio_path.stem}.wav"
    audio = AudioSegment.from_file(str(audio_path))
    audio.export(str(output_path), format="wav")
    return output_path

def add_speaker_profile():
    auth_token = "YOUR_TOKEN_HERE"  # Replace with your token
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(auth_token=auth_token)
    
    # Reference audio directory
    ref_dir = Path("data/reference_audio")
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files (not just WAV)
    audio_files = list(ref_dir.glob("*.mp3")) + list(ref_dir.glob("*.wav")) + list(ref_dir.glob("*.m4a"))
    
    if not audio_files:
        print("\nNo audio files found in reference_audio directory!")
        print("Please add some audio files (MP3 or WAV) to data/reference_audio/")
        return
        
    # Add each reference file
    print("\nAvailable reference files:")
    for i, file in enumerate(audio_files):
        print(f"{i+1}. {file.name}")
    
    file_num = input("\nSelect file number to add (or 'q' to quit): ")
    if file_num.lower() == 'q':
        return
    
    try:
        selected_file = audio_files[int(file_num) - 1]
        
        speaker_name = input("Enter speaker name: ")
        
        print(f"\nConverting audio to WAV format if needed...")
        wav_file = convert_to_wav(selected_file)
        
        print(f"Adding profile for {speaker_name} from {wav_file}")
        diarizer.add_speaker_profile(speaker_name, wav_file)
        
        # Save profiles
        profiles_file = Path("data/speaker_profiles.pkl")
        diarizer.speaker_identifier.save_profiles(profiles_file)
        print("Profile added and saved successfully!")
        
    except Exception as e:
        print(f"Error adding profile: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_speaker_profile()