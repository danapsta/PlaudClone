# test_db.py
from src.database.transcript_db import TranscriptDatabase, TranscriptEntry
from datetime import datetime
from pathlib import Path

def test_database():
    # Create test database
    db = TranscriptDatabase(Path("test.db"))
    
    # Create test entry
    entry = TranscriptEntry(
        file_name="test.mp3",
        timestamp=datetime.now(),
        full_text="Test transcript",
        speaker_segments="[]",
        summary=None
    )
    
    # Try to add entry
    print("Adding test entry to database...")
    db.add_transcript(entry)
    
    # Try to search
    print("Searching database...")
    results = db.search_transcripts("test")
    print(f"Found {len(results)} results")

if __name__ == "__main__":
    test_database()