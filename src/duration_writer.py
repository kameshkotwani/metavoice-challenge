import os
import json
import soundfile as sf
from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed
from config import AUDIO_DIR, JSON_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# CONFIGURATION
NUM_WORKERS = 8  # Number of parallel threads

# Ensure output directory exists

def get_file_duration(file_path:Path):
    try:
        with sf.SoundFile(file_path) as f:
            frames = len(f)
            samplerate = f.samplerate
            duration = frames / float(samplerate)
        return {"path": str(file_path.relative_to(AUDIO_DIR)), "duration": duration}
    except Exception as e:
        return {"path": str(file_path.relative_to(AUDIO_DIR)), "error": str(e)}


def process_folder(folder_path_str):
    folder_path = Path(folder_path_str)
    folder_name = folder_path.name
    records = []

    for file in folder_path.glob("*.flac"):
        record = get_file_duration(file)
        record["path"] = str(file.relative_to(AUDIO_DIR))  # Store relative path
        records.append(record)

    out_path = JSON_DIR / f"{folder_name}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return f"{folder_name} - ({len(records)} files)"

def main():
    all_folders = [f for f in AUDIO_DIR.iterdir() if f.is_dir()]
    print(f"Found {len(all_folders)} folders to process.")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_folder, str(folder)): folder.name
                for folder in all_folders
            }

            for future in as_completed(futures):
                folder = futures[future]
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error in {folder}: {e}")

if __name__ == "__main__":
    main()
