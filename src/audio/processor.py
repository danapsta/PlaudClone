# src/audio/processor.py
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import torch
from .transcriber import WhisperTranscriber
from .diarizer import SpeakerDiarizer

class AudioProcessor:
    def __init__(
        self,
        auth_token: str,
        whisper_model: str = "large",
        device: Optional[Union[str, torch.device]] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, str):
            device = torch.device(device)
            
        self.device = device
        
        # Initialize components
        self.transcriber = WhisperTranscriber(whisper_model, device)
        self.diarizer = SpeakerDiarizer(auth_token, device)

    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        """
        Process audio file with transcription and speaker diarization.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # Step 1: Transcribe audio
            self.logger.info("Starting transcription...")
            transcription = self.transcriber.transcribe(
                audio_path,
                language=language,
                preprocess=True
            )
            
            # Step 2: Perform speaker diarization
            self.logger.info("Starting speaker diarization...")
            speaker_segments = self.diarizer.diarize(audio_path)
            
            # Step 3: Combine results
            self.logger.info("Combining transcription with speaker segments...")
            transcript_segments = self.transcriber.get_segments(transcription)
            labeled_segments = self.diarizer.assign_transcription_to_segments(
                speaker_segments,
                transcript_segments
            )
            
            # Format results
            formatted_transcript = self.diarizer.format_transcript(labeled_segments)
            
            return {
                "full_transcript": transcription["text"],
                "speaker_segments": labeled_segments,
                "formatted_transcript": formatted_transcript
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {str(e)}")
            raise