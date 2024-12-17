# src/chat/transcript_query.py
from typing import List, Dict
from pathlib import Path
import sqlite3
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class TranscriptQueryEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        logger.info(f"Initialized TranscriptQueryEngine with database: {db_path}")

    def search_transcripts(self, query: str) -> List[Dict]:
        """Search transcripts for relevant content."""
        logger.info(f"Searching for: {query}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        file_name,
                        timestamp,
                        full_text,
                        speaker_segments
                    FROM transcripts
                    WHERE LOWER(full_text) LIKE LOWER(?)
                """, (f"%{query}%",))
                
                results = []
                for row in cursor:
                    speaker_segments = []
                    if row[3]:  # If speaker_segments is not None
                        try:
                            speaker_segments = json.loads(row[3])
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse speaker segments for {row[0]}")
                    
                    results.append({
                        'file_name': row[0],
                        'timestamp': datetime.fromisoformat(row[1]),
                        'text': row[2],
                        'speaker_segments': speaker_segments
                    })
                
                logger.info(f"Found {len(results)} matching transcripts")
                return results
                
        except Exception as e:
            logger.error(f"Database search failed: {e}", exc_info=True)
            raise

    def format_search_result(self, result: Dict) -> str:
        """Format a search result for display."""
        output = [
            f"File: {result['file_name']}",
            f"Date: {result['timestamp'].strftime('%Y-%m-%d %H:%M')}",
            "",
            "Transcript:"
        ]
        
        # If we have speaker segments, format them
        if result['speaker_segments']:
            for segment in result['speaker_segments']:
                speaker = segment.get('speaker', 'Unknown Speaker')
                text = segment.get('text', '')
                if text:
                    output.append(f"{speaker}: {text}")
        else:
            # If no speaker segments, just show the full text
            output.append(result['text'])
        
        output.append("-" * 80)
        return "\n".join(output)

class ChatInterface:
    def __init__(self, db_path: Path):
        self.query_engine = TranscriptQueryEngine(db_path)
        self.db_path = db_path
        logger.info("Chat interface initialized")

    def process_query(self, query: str) -> str:
        """Process a user query about transcripts."""
        try:
            # Special commands
            if query.lower() in ['list', 'list all', 'show all']:
                return self.list_all_transcripts()
                
            # Regular search
            results = self.query_engine.search_transcripts(query)
            
            if not results:
                return "I couldn't find any discussions about that topic in the transcripts."
            
            # Format response
            response = [f"I found {len(results)} relevant transcript(s):"]
            for result in results:
                response.append(self.query_engine.format_search_result(result))
            
            return "\n\n".join(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return "Sorry, there was an error processing your query. Please try again."

    def list_all_transcripts(self) -> str:
        """List all available transcripts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_name, timestamp
                    FROM transcripts
                    ORDER BY timestamp DESC
                """)
                
                results = cursor.fetchall()
                if not results:
                    return "No transcripts found in the database."
                
                response = ["Available transcripts:"]
                for file_name, timestamp in results:
                    dt = datetime.fromisoformat(timestamp)
                    response.append(f"- {file_name} ({dt.strftime('%Y-%m-%d %H:%M')})")
                
                return "\n".join(response)
                
        except Exception as e:
            logger.error(f"Error listing transcripts: {e}", exc_info=True)
            return "Sorry, there was an error retrieving the transcript list."