# check_files.py
from pathlib import Path
import os

def check_file_for_null_bytes(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                print(f"Found null bytes in {file_path}")
                return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return False

# Check all Python files in the project
project_root = Path('.')
python_files = list(project_root.rglob('*.py'))

print("\nFiles containing null bytes:")
null_byte_files = []
for file in python_files:
    if check_file_for_null_bytes(file):
        null_byte_files.append(file)

if not null_byte_files:
    print("No files with null bytes found!")

print("\nProject Structure (*.py files only):")
for root, dirs, files in os.walk('.'):
    if '.venv' in root or '__pycache__' in root:
        continue
    python_files = [f for f in files if f.endswith('.py')]
    if python_files:
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in python_files:
            print(f"{subindent}{f}")