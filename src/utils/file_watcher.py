# src/utils/file_watcher.py
from pathlib import Path
from typing import Set, Union
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, processor, output_dir: Path):
        self.processor = processor
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.processed_files: Set[Path] = set()
        
    def process_file(self, file_path: Path):
        try:
            # Process the audio file
            result = self.processor.process_audio(file_path, language="en")
            
            # Create output filename
            transcript_path = self.output_dir / f"{file_path.stem}_transcript.txt"
            
            # Save formatted transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["formatted_transcript"])
                
            self.logger.info(f"Processed and saved transcript: {transcript_path}")
            self.processed_files.add(file_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            return None

    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in {'.mp3', '.wav', '.m4a', '.flac'}:
            self.logger.info(f"New audio file detected: {file_path}")
            self.process_file(file_path)