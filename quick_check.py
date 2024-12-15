# quick_check.py
file_path = 'src/audio/transcriber.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"Successfully read {file_path}")
        print("First few characters:", repr(content[:50]))
except Exception as e:
    print(f"Error reading file: {e}")