# test_db_content.py
from pathlib import Path
import sqlite3

def check_database():
    db_path = Path("data/transcripts.db")
    print(f"Checking database at: {db_path.absolute()}")
    print(f"Database exists: {db_path.exists()}")
    
    if not db_path.exists():
        print("Database file not found!")
        return
        
    try:
        with sqlite3.connect(db_path) as conn:
            # Check if table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='transcripts'
            """)
            if not cursor.fetchone():
                print("No transcripts table found in database!")
                return
                
            # Count transcripts
            cursor = conn.execute("SELECT COUNT(*) FROM transcripts")
            count = cursor.fetchone()[0]
            print(f"\nFound {count} transcripts in database")
            
            # Show sample entries
            cursor = conn.execute("""
                SELECT file_name, timestamp, full_text 
                FROM transcripts 
                LIMIT 3
            """)
            print("\nSample entries:")
            for row in cursor:
                print(f"\nFile: {row[0]}")
                print(f"Time: {row[1]}")
                print(f"Text preview: {row[2][:100]}...")
                
    except Exception as e:
        print(f"Error accessing database: {e}")

if __name__ == "__main__":
    check_database()