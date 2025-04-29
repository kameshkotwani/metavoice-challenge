from pathlib import Path

# Define the project root as the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
# Define the data root as a subdirectory named 'data' within the project root
RAW_DATA_DIR = DATA_DIR / 'raw'
AUDIO_DIR = RAW_DATA_DIR / 'wav48_silence_trimmed'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
JSON_DIR = PROCESSED_DATA_DIR / 'jsonl'
TEST_DATA_DIR = DATA_DIR / 'test'
RESULTS_DIR = PROJECT_ROOT / 'results'

print(PROJECT_ROOT)