# src/audio/diarizer.py
import os
from pyannote.audio import Pipeline
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import logging
from dataclasses import dataclass
import numpy as np

# Set environment Variable to disable symLinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a single speaker"""
    speaker: str
    start: float
    end: float
    text: Optional[str] = None

class SpeakerDiarizer:
    def __init__(self, 
                 auth_token: str, 
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the speaker diarization pipeline.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Convert string to torch.device if necessary
        if isinstance(device, str):
            device = torch.device(device)
            
        self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.logger.info("Loading diarization pipeline...")
            # Use local cache directory
            cache_dir = Path("models/pyannote").absolute()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token,
                cache_dir=cache_dir
            ).to(self.device)
            self.logger.info("Diarization pipeline loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load diarization pipeline: {str(e)}")
            raise

    def diarize(self, audio_path: Path) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of SpeakerSegment objects
        """
        try:
            self.logger.info(f"Starting diarization for: {audio_path}")
            
            # Run diarization
            diarization = self.pipeline(str(audio_path))
            
            # Convert results to speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker=f"SPEAKER_{speaker.split('#')[-1]}",
                    start=turn.start,
                    end=turn.end
                )
                segments.append(segment)
            
            self.logger.info(f"Diarization completed. Found {len(set(s.speaker for s in segments))} speakers")
            return segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed for {audio_path}: {str(e)}")
            raise

    def assign_transcription_to_segments(
        self,
        segments: List[SpeakerSegment],
        transcription_segments: List[Dict]
    ) -> List[SpeakerSegment]:
        """
        Assign transcribed text to speaker segments based on timestamp overlap.
        
        Args:
            segments: List of speaker segments
            transcription_segments: List of transcription segments with timestamps
            
        Returns:
            Updated list of speaker segments with text
        """
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_text = trans_seg["text"]
            
            # Find overlapping speaker segments
            for segment in segments:
                # Check for significant overlap
                overlap_start = max(segment.start, trans_start)
                overlap_end = min(segment.end, trans_end)
                overlap = overlap_end - overlap_start
                
                if overlap > 0:
                    # If segment has no text, assign it
                    if segment.text is None:
                        segment.text = trans_text
                    # If segment already has text, append new text
                    else:
                        segment.text += " " + trans_text
        
        # Remove segments without text
        segments = [s for s in segments if s.text is not None]
        
        return segments

    def format_transcript(self, segments: List[SpeakerSegment]) -> str:
        """Format speaker segments into a readable transcript."""
        transcript = []
        for segment in segments:
            timestamp = f"[{segment.start:.1f}s - {segment.end:.1f}s]"
            transcript.append(f"{segment.speaker} {timestamp}: {segment.text}")
        return "\n".join(transcript)