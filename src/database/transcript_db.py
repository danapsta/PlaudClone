# src/database/transcript_db.py
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from datetime import datetime
import json
from typing import List, Optional, Union

@dataclass
class TranscriptEntry:
    file_name: str
    timestamp: datetime
    full_text: str
    speaker_segments: str  # JSON string of segments
    summary: Optional[str] = None

class TranscriptDatabase:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.init_db()
        
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    full_text TEXT NOT NULL,
                    speaker_segments TEXT NOT NULL,
                    summary TEXT
                )
            """)
            
    def add_transcript(self, entry: TranscriptEntry):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO transcripts
                (file_name, timestamp, full_text, speaker_segments, summary)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entry.file_name,
                entry.timestamp.isoformat(),
                entry.full_text,
                entry.speaker_segments,
                entry.summary
            ))
            
    def search_transcripts(self, query: str) -> List[TranscriptEntry]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM transcripts
                WHERE full_text LIKE ? OR summary LIKE ?
            """, (f"%{query}%", f"%{query}%"))
            
            return [TranscriptEntry(
                file_name=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                full_text=row[3],
                speaker_segments=row[4],
                summary=row[5]
            ) for row in cursor.fetchall()]