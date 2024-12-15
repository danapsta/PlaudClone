# test_simple.py
try:
    print("Attempting import...")
    from src.audio.transcriber import WhisperTranscriber
    print("Import successful!")
    print("Creating transcriber instance...")
    transcriber = WhisperTranscriber()
    print("Transcriber created successfully!")
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"Error message: {str(e)}")