# src/audio/transcriber.py
import whisper
import torch
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import numpy as np
import soundfile as sf
from scipy import signal

class AudioPreprocessor:
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to -20dB"""
        target_dB = -20
        current_dB = 20 * np.log10(np.sqrt(np.mean(audio**2)))
        adjustment = target_dB - current_dB
        return audio * (10 ** (adjustment / 20))

    @staticmethod
    def remove_noise(audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple noise reduction using a high-pass filter"""
        # Remove very low frequencies (below 100Hz)
        nyquist = sr // 2
        cutoff = 100 / nyquist
        b, a = signal.butter(4, cutoff, btype='high')
        return signal.filtfilt(b, a, audio)

class WhisperTranscriber:
    def __init__(self, model_name: str = "large", device: Optional[str] = None):
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

    def preprocess_audio(self, audio_path: Path) -> Path:
        """Preprocess audio file and return path to processed version"""
        try:
            self.logger.info("Starting audio preprocessing...")
            
            # Read audio file
            audio, sr = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Apply preprocessing
            audio = AudioPreprocessor.remove_noise(audio, sr)
            audio = AudioPreprocessor.normalize_audio(audio)
            
            # Save processed audio
            processed_path = audio_path.parent / f"processed_{audio_path.name}"
            sf.write(str(processed_path), audio, sr)
            
            self.logger.info("Audio preprocessing completed")
            return processed_path
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            self.logger.warning("Falling back to original audio file")
            return audio_path

    def transcribe(self, 
                  audio_path: Union[str, Path], 
                  language: Optional[str] = None,
                  preprocess: bool = True,
                  **kwargs) -> Dict:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en" for English)
            preprocess: Whether to apply audio preprocessing
            **kwargs: Additional arguments to pass to whisper.transcribe
            
        Returns:
            Dictionary containing transcription results
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            self.logger.info(f"Starting transcription of: {audio_path}")
            
            # Preprocess audio if requested
            if preprocess:
                audio_path = self.preprocess_audio(audio_path)
            
            # Set default transcription options
            options = {
                "language": language,
                "task": "transcribe",
                "verbose": True,
                "best_of": 5,  # Increased beam search
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Temperature search
                "compression_ratio_threshold": 2.4,
                "condition_on_previous_text": True,
            }
            options.update(kwargs)
            
            # Perform transcription
            result = self.model.transcribe(str(audio_path), **options)
            
            self.logger.info(f"Transcription completed for: {audio_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path}: {str(e)}")
            raise
        finally:
            # Clean up processed file if it exists
            if preprocess and 'processed_' in str(audio_path):
                try:
                    audio_path.unlink()
                except:
                    pass

    def get_segments(self, transcription: Dict) -> list:
        """Extract segments with timestamps from transcription."""
        return [
            {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            }
            for segment in transcription["segments"]
        ]