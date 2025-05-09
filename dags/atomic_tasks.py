from airflow import DAG
from airflow.decorators import task
from datetime import timedelta
from pathlib import Path
import json
import torch
import numpy as np
import soundfile as sf
import requests
from filelock import FileLock
from config import DATA_DIR,RAW_DATA_DIR,TEST_DATA_DIR,JSON_DIR,RESULTS_DIR

# Config paths
AUDIO_ROOT = TEST_DATA_DIR
TEMP_JSON_DIR = RESULTS_DIR / "transcriptions"
OUTPUT_FILE = RESULTS_DIR / "all_results.jsonl"

LOCK_FILE = OUTPUT_FILE.with_suffix(".lock")

TRANSCRIBE_API_URL = "http://localhost:8000/transcribe"  

# === TOKENISATION FUNCTION (Provided Mock) ===
def tokenise(audio_np_array: np.ndarray) -> torch.Tensor:
    import time
    if not isinstance(audio_np_array, np.ndarray):
        raise ValueError("Input should be a NumPy array")
    del audio_np_array
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available.")

    start_time = time.time()
    while True:
        tensor_length = np.random.randint(20, 1001)
        result_tensor = torch.randint(low=-32768, high=32767, size=(tensor_length,), dtype=torch.int16, device='cuda')
        a = torch.rand(5000, 5000, device='cuda')
        b = torch.rand(5000, 5000, device='cuda')
        _ = torch.matmul(a, b)
        if (time.time() - start_time) * 1000 >= 200:
            break
    return result_tensor

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="audio_pipeline_split_atomic",
    default_args=default_args,
    catchup=False,
    max_active_tasks=10,
    tags=["atomic", "whisper", "tokenise","fastapi-server","gpu_pool,default_pool"],
) as dag:

    @task(pool="default_pool",retries=2, retry_delay=timedelta(minutes=1)) 
    def list_audio_files():
        return [
            str(p.relative_to(AUDIO_ROOT))
            for p in AUDIO_ROOT.rglob("*.flac")
        ]

    @task(pool="default_pool",retries=2, retry_delay=timedelta(minutes=1)) 
    def truncate_jsonl_files():
        results_path = RESULTS_DIR  
        deleted_files = 0

        for file in results_path.glob("*.jsonl"):
            try:
                with open(file, "w") as f:
                    pass  # Opening with "w" mode truncates the file to zero length
                deleted_files += 1
            except Exception as e:
                print(f"Failed to truncate {file.name}: {e}")

        if deleted_files == 0:
            print("No .jsonl files found to truncate.")
        else:
            print(f"Truncated {deleted_files} JSONL files.")

        return deleted_files
    
    @task(pool="default_pool",retries=2, retry_delay=timedelta(minutes=1))  # use CPU pool for transcription
    def transcribe_file(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        temp_path = TEMP_JSON_DIR / f"{file_id}.json"

        if temp_path.exists():
            return f"Skipped {file_id}"  # already processed

        res = requests.post(TRANSCRIBE_API_URL, json={"file_path": file_relative_path})
        res.raise_for_status()
        transcription = res.json()["transcription"].strip()

        with open(temp_path, "w") as f:
            json.dump({
                "id": file_relative_path,
                "transcription": transcription
            }, f)

        return f"Saved {file_id}"

    @task(pool="gpu_pool",retries=2, retry_delay=timedelta(minutes=1))
    def tokenise_file(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        temp_path = TEMP_JSON_DIR / f"{file_id}.json"

        if not temp_path.exists():
            raise FileNotFoundError(f"Missing transcription for {file_relative_path}")

        # Load transcription
        with open(temp_path) as f:
            partial = json.load(f)

        # Load audio and tokenise
        audio_np, _ = sf.read(AUDIO_ROOT / file_relative_path, dtype="float32")
        token_tensor = tokenise(audio_np)
        token_list = token_tensor.tolist()
        del token_tensor
        torch.cuda.empty_cache()

        # === Determine output based on FOLDER ===
        folder_name = Path(file_relative_path).parts[0]  # 'p225' or 'p226'
        output_file = RESULTS_DIR / f"{folder_name}.jsonl"
        lock_file = output_file.with_suffix(".lock")

        partial["token_array"] = token_list

        # Safe write to the correct per-folder JSONL
        with FileLock(str(lock_file)):
            with open(output_file, "a") as out:
                out.write(json.dumps(partial) + "\n")
        
        del token_list
        return f"Written {file_id} to {folder_name}.jsonl"

    @task()
    def mark_tokenisation_complete():
        return "Tokenisation completed"

    # removing the intermediate json files
    @task(pool="default_pool",retries=2, retry_delay=timedelta(minutes=1))
    def cleanup_intermediate_jsons():
        import os

        deleted = 0
        for f in TEMP_JSON_DIR.glob("*.json"):
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}")
        return f"Deleted {deleted} temp transcription files"


    # audio_files = list_audio_files()
    # transcriptions = transcribe_file.expand(file_relative_path=audio_files)
    # tokenisations = tokenise_file.expand(file_relative_path=audio_files)
    # tokenisations.set_upstream(transcriptions)

    # cleanup_intermediate_jsons().set_upstream(tokenise_file.expand(file_relative_path=audio_files))
    # audio_files = list_audio_files()

    # truncate = truncate_jsonl_files()
    # # Mapped tasks
    # transcriptions = transcribe_file.expand(file_relative_path=audio_files)
    # tokenisations = tokenise_file.expand(file_relative_path=audio_files)

    # # Dummy marker for sequencing
    # done = mark_tokenisation_complete()

    # # Cleanup task
    # cleanup = cleanup_intermediate_jsons()

    # # Chained execution
    # audio_files >> truncate >> transcriptions >> tokenisations >> done >> cleanup


    # List audio files
    audio_files = list_audio_files()

    # Expand mapped tasks
    transcriptions = transcribe_file.expand(file_relative_path=audio_files)
    tokenisations = tokenise_file.expand(file_relative_path=audio_files)

    # Chain mapped tokenisations to transcriptions
    tokenisations.set_upstream(transcriptions)

    # Cleanup after tokenisation (with dummy marker)
    done = mark_tokenisation_complete()
    done.set_upstream(tokenisations)

    cleanup_intermediate_jsons().set_upstream(done)

    # Optional: truncate JSONL files before everything
    truncate_jsonl_files().set_upstream(audio_files)