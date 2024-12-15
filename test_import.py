# test_import.py
try:
    from src.audio.transcriber import WhisperTranscriber
    print("Successfully imported WhisperTranscriber")
except Exception as e:
    print(f"Import failed with error: {type(e).__name__}")
    print(f"Error message: {str(e)}")