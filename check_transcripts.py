# check_transcripts.py
import sqlite3
from pathlib import Path

def check_database():
    db_path = Path("data/transcripts.db")
    print(f"Checking database at: {db_path.absolute()}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get table info
            cursor = conn.execute("SELECT * FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print("\nDatabase tables:")
            for table in tables:
                print(f"- {table[1]}")
            
            # Get transcript count
            cursor = conn.execute("SELECT COUNT(*) FROM transcripts")
            count = cursor.fetchone()[0]
            print(f"\nNumber of transcripts: {count}")
            
            # Show all transcripts
            cursor = conn.execute("""
                SELECT file_name, timestamp, full_text, speaker_segments 
                FROM transcripts
            """)
            print("\nAll transcripts:")
            print("-" * 80)
            for row in cursor:
                print(f"\nFile: {row[0]}")
                print(f"Time: {row[1]}")
                print(f"Text length: {len(row[2]) if row[2] else 0}")
                print(f"Has speaker segments: {'Yes' if row[3] else 'No'}")
                print(f"First 100 chars: {row[2][:100] if row[2] else 'None'}...")
                print("-" * 40)
                
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_database()