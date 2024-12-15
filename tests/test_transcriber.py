# tests/test_transcriber.py
import pytest
from pathlib import Path
from src.audio.transcriber import WhisperTranscriber

def test_transcriber_initialization():
    transcriber = WhisperTranscriber(model_name="base")
    assert transcriber.device in ["cuda", "cpu"]
    assert transcriber.model is not None

def test_transcribe_file_not_found():
    transcriber = WhisperTranscriber(model_name="base")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("nonexistent_file.mp3")