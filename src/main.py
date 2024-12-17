# src/main.py
from pathlib import Path
import logging
import time
import json  # Add this import
from watchdog.observers import Observer
from .utils.file_watcher import AudioFileHandler
from .audio.processor import AudioProcessor
from .database.transcript_db import TranscriptDatabase, TranscriptEntry
from datetime import datetime

# src/main.py
class TranscriptionSystem:
    def __init__(self, auth_token: str):
        # Setup directories
        self.base_dir = Path(__file__).parent.parent
        self.watch_dir = self.base_dir / "data" / "audio"
        self.output_dir = self.base_dir / "data" / "transcripts"  # Changed this line
        self.db_path = self.base_dir / "data" / "transcripts.db"
        
        # Create directories if they don't exist
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = AudioProcessor(auth_token=auth_token)
        self.db = TranscriptDatabase(self.db_path)
        
        # Setup file watcher
        self.handler = AudioFileHandler(self.processor, self.output_dir)
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        
    def start(self):
        """Start watching for new files"""
        self.observer.start()
        logging.info(f"Started watching directory: {self.watch_dir}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            self.observer.join()
            
    def process_existing_files(self):
        """Process any existing files in the watch directory"""
        def segment_to_dict(segment):
            """Convert SpeakerSegment to dictionary"""
            return {
                'speaker': segment.speaker,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'confidence': segment.confidence
            }

        for file_path in self.watch_dir.glob("*"):
            if file_path.suffix.lower() in {'.mp3', '.wav', '.m4a', '.flac'}:
                if file_path not in self.handler.processed_files:
                    logging.info(f"Processing existing file: {file_path}")
                    result = self.handler.process_file(file_path)
                    if result:
                        # Convert speaker segments to JSON-serializable format
                        serializable_segments = [
                            segment_to_dict(segment) 
                            for segment in result["speaker_segments"]
                        ]
                    
                        # Add to database
                        entry = TranscriptEntry(
                            file_name=file_path.name,
                            timestamp=datetime.now(),
                            full_text=result["full_transcript"],
                            speaker_segments=json.dumps(serializable_segments),
                        )
                        self.db.add_transcript(entry)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Your HuggingFace token
    auth_token = "YOUR_TOKEN_HERE"
    
    system = TranscriptionSystem(auth_token)
    
    # Process any existing files first
    system.process_existing_files()
    
    # Start watching for new files
    system.start()

if __name__ == "__main__":
    main()