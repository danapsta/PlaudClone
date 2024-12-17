# test_profile_content.py
from pathlib import Path
import pickle

def test_profiles():
    profile_path = Path("data/speaker_profiles.pkl")
    print(f"Looking for profiles at: {profile_path.absolute()}")
    print(f"File exists: {profile_path.exists()}")

    if not profile_path.exists():
        print("No profile file found!")
        return

    try:
        with open(profile_path, 'rb') as f:
            profiles = pickle.load(f)

        print("\nLoaded profiles:")
        for name, profile in profiles.items():
            print(f"\nSpeaker: {name}")
            print(f"Number of samples: {len(profile.embeddings)}")
            print(f"Sample paths: {profile.audio_samples}")
            if profile.embeddings:
                print(f"First embedding type: {type(profile.embeddings[0])}")
                if hasattr(profile.embeddings[0], 'shape'):
                    print(f"Embedding shape: {profile.embeddings[0].shape}")
    except Exception as e:
        print(f"Error loading profiles: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_profiles()