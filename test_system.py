# test_system.py
from src.main import TranscriptionSystem
import logging
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_transcription_system():
    # Your HuggingFace token
    auth_token = "YOUR_TOKEN_HERE"  # Replace with your token
    
    print("\nInitializing Transcription System...")
    system = TranscriptionSystem(auth_token)
    
    # Print directory information
    print("\nDirectory Setup:")
    print(f"Watching: {system.watch_dir}")
    print(f"Saving transcripts to: {system.output_dir}")
    print(f"Database location: {system.db_path}")
    
    # Process existing files
    print("\nProcessing any existing files...")
    system.process_existing_files()
    
    # Print results
    print("\nResults:")
    if system.handler.processed_files:
        print("\nProcessed files:")
        for file in system.handler.processed_files:
            print(f"- {file}")
    else:
        print("No files processed")
    
    # List generated transcripts
    print("\nGenerated transcripts:")
    transcripts = list(system.output_dir.glob("*_transcript.txt"))
    if transcripts:
        for transcript in transcripts:
            print(f"- {transcript.name}")
            # Print first few lines of transcript
            with open(transcript, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:5])
                print(f"Preview:\n{first_lines}")
            print("-" * 80)
    else:
        print("No transcripts generated")

if __name__ == "__main__":
    test_transcription_system()