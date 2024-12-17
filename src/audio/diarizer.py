# src/audio/diarizer.py
import os
from pyannote.audio import Pipeline
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import logging
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from .speaker_identity import SpeakerIdentifier
from pydub import AudioSegment
import tempfile
import os

# Set environment Variable to disable symLinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a single speaker"""
    speaker: str
    start: float
    end: float
    text: Optional[str] = None
    confidence: Optional[float] = None


class SpeakerDiarizer:
    def __init__(self, auth_token: str, device: Optional[torch.device] = None):
        self.logger = logging.getLogger(__name__)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

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

            self.speaker_identifier = SpeakerIdentifier(auth_token, device)
            self.logger.info("Diarization pipeline loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load diarization pipeline: {str(e)}")
            raise

    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert audio file to WAV format."""
        try:
            self.logger.info(f"Converting {audio_path} to WAV format...")

            # Load Audio File
            audio = AudioSegment.from_file(str(audio_path))
            self.logger.info(
                f"Successfully loaded audio file: {len(audio)}ms duration")

            # Create temporary WAV file
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            wav_path = temp_dir / f"{audio_path.stem}_temp.wav"

            # Export as Wav
            audio.export(wav_path, format="wav")
            self.logger.info(f"Successfully exported to: {wav_path}")

            return wav_path
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            raise

    def load_speaker_profiles(self, path: Path) -> None:
        """Load known speaker profiles."""
        self.speaker_identifier.load_profiles(path)

    def add_speaker_profile(self, name: str, audio_path: Path) -> None:
        """Add a new speaker profile."""
        self.speaker_identifier.add_speaker(name, audio_path)

    def diarize(self, audio_path: Path) -> List[SpeakerSegment]:
        """Perform speaker diarization on an audio file."""
        try:
            self.logger.info(f"Starting diarization for: {audio_path}")

            # Convert to WAV if needed
            if audio_path.suffix.lower() != '.wav':
                self.logger.info("Converting audio to WAV format...")
                wav_path = self._convert_to_wav(audio_path)
                self.logger.info(f"Audio converted: {wav_path}")
            else:
                wav_path = audio_path

            # Run diarization
            diarization = self.pipeline(str(wav_path))

            # Load audio for speaker identification
            audio, sample_rate = sf.read(str(wav_path))

            # Convert results to speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Extract audio segment
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                segment_audio = audio[start_sample:end_sample]

                # Default speaker label
                speaker_label = f"SPEAKER_{speaker.split('#')[-1]}"
                confidence = 0.0

                # Try to identify the speaker
                if hasattr(self, 'speaker_identifier'):
                    try:
                        identified_speaker, conf = self.speaker_identifier.identify_speaker(
                            segment_audio)
                        if identified_speaker:
                            speaker_label = identified_speaker
                            confidence = conf
                    except Exception as e:
                        self.logger.warning(
                            f"Speaker identification failed for segment: {e}")

                segment = SpeakerSegment(
                    speaker=speaker_label,
                    start=turn.start,
                    end=turn.end,
                    confidence=confidence
                )
                segments.append(segment)

            # Clean up temporary file
            if audio_path.suffix.lower() != '.wav':
                try:
                    wav_path.unlink()
                except:
                    pass

            self.logger.info(
                f"Diarization completed. Found {len(set(s.speaker for s in segments))} speakers")
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
            confidence_str = f"({segment.confidence:.1%})" if segment.confidence > 0 else "(Unknown)"
            transcript.append(
                f"{segment.speaker} {confidence_str} {timestamp}: {segment.text}")
        return "\n".join(transcript)
