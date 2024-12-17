from typing import List, Dict, Optional
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import logging
import requests
import subprocess
import time

class TranscriptQuery:
    def __init__(self, db_path: Path, model_name: str = "llama2:3.2"):
        """
        Initialize the transcript query system using Ollama.
        
        Args:
            db_path: Path to the transcript database
            model_name: Name of the Ollama model to use (default: "llama2:3.2")
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Check available models and validate model name
        self.model_name = self._validate_model(model_name)
        
        # Try different Ollama API endpoints
        self.base_urls = [
            "http://localhost:11434",
            "http://127.0.0.1:11434"
        ]
        
        # Test connection
        self.api_url = self._get_working_endpoint()
        if not self.api_url:
            raise ConnectionError("Could not connect to Ollama API")

    def _validate_model(self, requested_model: str) -> str:
        """Validate and return correct model name."""
        try:
            # Get list of available models
            result = subprocess.run(['ollama', 'list'], 
                                 capture_output=True, 
                                 text=True)
            
            available_models = result.stdout.split('\n')
            self.logger.info(f"Available models: {available_models}")
            
            # Check if requested model is available
            if any(requested_model in model for model in available_models):
                return requested_model
            
            # If not found, try to pull it
            self.logger.info(f"Model {requested_model} not found, attempting to pull...")
            subprocess.run(['ollama', 'pull', requested_model], 
                         check=True)
            return requested_model
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error validating model: {str(e)}")
            # Fall back to a default model if available
            if any('llama2' in model for model in available_models):
                self.logger.info("Falling back to llama2 model")
                return "llama2"
            raise ValueError("No suitable model found")

    def _get_working_endpoint(self) -> Optional[str]:
        """Try different endpoints to find one that works."""
        for base_url in self.base_urls:
            try:
                response = requests.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    self.logger.info(f"Successfully connected to Ollama at {base_url}")
                    return base_url
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to connect to {base_url}: {str(e)}")
        return None

    def _query_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Send a query to Ollama and get the response."""
        if not self.api_url:
            raise ConnectionError("No working Ollama API endpoint found")
            
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 404:
                    self.logger.error("API endpoint not found. Is Ollama running?")
                    raise ConnectionError("Ollama API endpoint not found")
                    
                response.raise_for_status()
                return response.json()['response']
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise ConnectionError(f"Failed to connect to Ollama after {max_retries} attempts")

    def _fetch_transcripts(self, query: Optional[str] = None) -> List[Dict]:
        """Fetch transcripts from database with optional search query."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if query:
                    sql = """
                        SELECT * FROM transcripts 
                        WHERE full_text LIKE ? OR summary LIKE ?
                    """
                    cursor = conn.execute(sql, (f"%{query}%", f"%{query}%"))
                else:
                    cursor = conn.execute("SELECT * FROM transcripts")
                
                results = []
                for row in cursor.fetchall():
                    # Parse speaker segments from JSON string
                    segments = json.loads(row[4])  # speaker_segments column
                    
                    results.append({
                        "file_name": row[1],
                        "timestamp": datetime.fromisoformat(row[2]),
                        "full_text": row[3],
                        "speaker_segments": segments,
                        "summary": row[5]
                    })
                
                return results
        except Exception as e:
            self.logger.error(f"Database query failed: {str(e)}")
            raise

    def _create_conversation_context(self, transcripts: List[Dict]) -> str:
        """Create a context string from transcripts for the LLM."""
        context = "Here are the relevant conversation transcripts:\n\n"
        
        for transcript in transcripts:
            context += f"Conversation from {transcript['timestamp']}:\n"
            context += "Speakers and their statements:\n"
            
            for segment in transcript['speaker_segments']:
                context += f"{segment['speaker']}: {segment['text']}\n"
            
            context += "\n---\n\n"
            
        return context

    def _extract_action_items(self, text: str) -> List[Dict]:
        """Extract action items from text using Ollama."""
        prompt = f"""
        Given the following conversation text, identify any action items or tasks that were mentioned or implied.
        For each action item, determine:
        1. What needs to be done
        2. Who should do it (if mentioned)
        3. Any deadline or timeline mentioned
        4. The priority level (High/Medium/Low) based on context

        Return the response in this exact JSON format:
        [
            {{
                "task": "Task description",
                "assignee": "Person name or 'Unspecified'",
                "deadline": "Deadline or 'None specified'",
                "priority": "High/Medium/Low"
            }}
        ]

        Conversation text:
        {text}
        """
        
        try:
            response = self._query_ollama(prompt)
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return []
        except Exception as e:
            self.logger.error(f"Failed to parse action items: {str(e)}")
            return []

    def query_transcripts(self, user_query: str) -> str:
        """
        Query transcripts and generate a response using Ollama.
        
        Args:
            user_query: User's question or query
            
        Returns:
            Generated response from Ollama
        """
        # Fetch relevant transcripts
        transcripts = self._fetch_transcripts()
        
        # Create context for LLM
        context = self._create_conversation_context(transcripts)
        
        # Create the prompt
        prompt = f"""
        You are a helpful AI assistant analyzing conversation transcripts. 
        Use the following conversation context to answer the user's question.
        Only use information that is explicitly present in the conversations.
        If you're not sure about something, say so.

        {context}

        User question: {user_query}

        Please provide a clear and concise answer:
        """
        
        return self._query_ollama(prompt)

    def get_action_items(self) -> List[Dict]:
        """Get all action items from transcripts."""
        transcripts = self._fetch_transcripts()
        all_actions = []
        
        for transcript in transcripts:
            actions = self._extract_action_items(transcript['full_text'])
            # Add source information
            for action in actions:
                action['source'] = {
                    'file': transcript['file_name'],
                    'date': transcript['timestamp'].isoformat()
                }
            all_actions.extend(actions)
            
        return all_actions

    def get_speaker_summary(self, speaker_name: str) -> Dict:
        """Get a summary of a specific speaker's contributions."""
        transcripts = self._fetch_transcripts()
        
        # Collect all statements by the speaker
        statements = []
        for transcript in transcripts:
            for segment in transcript['speaker_segments']:
                if segment['speaker'].lower() == speaker_name.lower():
                    statements.append({
                        'text': segment['text'],
                        'date': transcript['timestamp'],
                        'file': transcript['file_name']
                    })
        
        if not statements:
            return {"error": f"No statements found for speaker {speaker_name}"}
            
        # Create prompt for summary
        statements_text = "\n".join([s['text'] for s in statements])
        prompt = f"""
        Analyze these statements by {speaker_name} and provide a response in this exact JSON format:
        {{
            "main_topics": ["topic1", "topic2", ...],
            "key_points": ["point1", "point2", ...],
            "communication_patterns": ["pattern1", "pattern2", ...]
        }}

        Statements:
        {statements_text}
        """
        
        try:
            response = self._query_ollama(prompt)
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                summary = json.loads(json_str)
                summary['total_statements'] = len(statements)
                summary['date_range'] = {
                    'start': min(s['date'] for s in statements).isoformat(),
                    'end': max(s['date'] for s in statements).isoformat()
                }
                return summary
            return {"error": "Failed to parse summary"}
        except Exception as e:
            self.logger.error(f"Failed to generate speaker summary: {str(e)}")
            return {"error": "Failed to generate summary"}