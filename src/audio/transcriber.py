import whisper
import torch
from pathlib import Path
from typing import Dict, Optional, Union
import logging

class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize the WhisperTranscriber with specified model.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name).to(self.device)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe(self, 
                  audio_path: Union[str, Path], 
                  language: Optional[str] = None,
                  **kwargs) -> Dict:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en" for English)
            **kwargs: Additional arguments to pass to whisper.transcribe
            
        Returns:
            Dictionary containing transcription results
        """
        audio_path = Path(audio_path).resolve() # Get absolute path
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            self.logger.info(f"Starting transcription of: {audio_path}")
            
            # Convert path to string and ensure it's absolute
            audio_path_str = str(audio_path.absolute())

            # Print debugging info
            self.logger.info(f"File exists check: {Path(audio_path_str).exists()}")
            self.logger.info(f"File size: {audio_path.stat().st_size} bytes")

            # Set default transcription options
            options = {
                "language": language,
                "task": "transcribe",
                "verbose": False
            }
            options.update(kwargs)
            
            # Perform transcription
            result = self.model.transcribe(str(audio_path), **options)
            
            self.logger.info(f"Transcription completed for: {audio_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path}: {str(e)}")
            raise

    def get_segments(self, transcription: Dict) -> list:
        """
        Extract segments with timestamps from transcription.
        
        Args:
            transcription: Transcription result dictionary
            
        Returns:
            List of segments with text and timestamps
        """
        return [
            {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            }
            for segment in transcription["segments"]
        ]