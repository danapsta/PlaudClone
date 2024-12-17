# add_speaker_profile.py
from src.audio.diarizer import SpeakerDiarizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_speaker_profile():
    auth_token = "YOUR_TOKEN_HERE"  # Replace with your token
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(auth_token=auth_token)
    
    # Reference audio directory
    ref_dir = Path("data/reference_audio")
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    # Add each reference file
    print("\nAvailable reference files:")
    for i, file in enumerate(ref_dir.glob("*.wav")):
        print(f"{i+1}. {file.name}")
    
    file_num = input("\nSelect file number to add (or 'q' to quit): ")
    if file_num.lower() == 'q':
        return
    
    try:
        files = list(ref_dir.glob("*.wav"))
        selected_file = files[int(file_num) - 1]
        
        speaker_name = input("Enter speaker name: ")
        
        print(f"\nAdding profile for {speaker_name} from {selected_file}")
        diarizer.add_speaker_profile(speaker_name, selected_file)
        
        # Save profiles
        profiles_file = Path("data/speaker_profiles.pkl")
        diarizer.speaker_identifier.save_profiles(profiles_file)
        print("Profile added and saved successfully!")
        
    except Exception as e:
        print(f"Error adding profile: {e}")

if __name__ == "__main__":
    add_speaker_profile()