from src.audio.transcriber import WhisperTranscriber
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def test_transcription(audio_file_path: str | Path):
    # Convert string path to Path object
    audio_path = Path(audio_file_path)
    
    # Initialize transcriber with medium model
    transcriber = WhisperTranscriber(model_name="medium")
    
    # Perform transcription
    try:
        result = transcriber.transcribe(
            audio_path,
            language="en",  # Specify language if known
            fp16=False  # Use this if you encounter CUDA errors
        )
        
        # Print full transcription
        print("\nFull Transcription:")
        print("------------------")
        print(result["text"])
        
        # Print segments with timestamps
        print("\nSegmented Transcription:")
        print("----------------------")
        segments = transcriber.get_segments(result)
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            print(f"[{start:.2f}s - {end:.2f}s]: {text}")
            
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Using pathlib to construct the path
    project_root = Path(__file__).parent
    audio_file = project_root / "data" / "audio" / "12-13 Meeting_ Evening Routine, Hygiene, and Safety Rules.mp3"
    
    print(f"Attempting to transcribe: {audio_file}")
    print(f"File exists: {audio_file.exists()}")
    
    test_transcription(audio_file)