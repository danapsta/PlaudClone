# src/utils/file_watcher.py
from pathlib import Path
from typing import Set
from typing import Set, Union
import time
import logging

class AudioFileWatcher:
    def __init__(self, watch_dir: Union[str, Path]):
        self.watch_dir = Path(watch_dir)
        self.logger = logging.getLogger(__name__)
        self.processed_files: Set[Path] = set()
        
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True)
            self.logger.info(f"Created watch directory: {self.watch_dir}")
    
    def get_new_files(self) -> Set[Path]:
        """Get new audio files that haven't been processed yet."""
        current_files = set(
            p for p in self.watch_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
        )
        new_files = current_files - self.processed_files
        return new_files
    
    def mark_as_processed(self, file_path: Path):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
