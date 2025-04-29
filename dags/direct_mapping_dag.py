from airflow import DAG
from airflow.decorators import task
from datetime import timedelta, datetime
from pathlib import Path
import json
import torch
import numpy as np
import soundfile as sf
import requests
from filelock import FileLock
from config import TEST_DATA_DIR, RESULTS_DIR

# Paths
AUDIO_ROOT = TEST_DATA_DIR
TEMP_JSON_DIR = RESULTS_DIR / "transcriptions"
TEMP_JSON_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRANSCRIBE_API_URL = "http://localhost:8000/transcribe"

# === Tokenisation Mock ===
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
    dag_id="audio_pipeline_direct_mapping",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_tasks=20,
    tags=["no_xcom", "parallel", "safe", "fastapi", "atomic"],
) as dag:

    @task(pool="default_pool")
    def prepare_jsonl_files():
        folders = {p.parts[0] for p in AUDIO_ROOT.rglob("*.flac")}
        created, truncated = 0, 0

        for folder in folders:
            jsonl_path = RESULTS_DIR / f"{folder}.jsonl"
            try:
                with open(jsonl_path, "w") as f:
                    pass  # This truncates the file if it exists
                if jsonl_path.exists():
                    truncated += 1
                else:
                    created += 1
            except Exception as e:
                print(f"Failed to create/truncate {jsonl_path}: {e}")

        print(f"Truncated {truncated} and created {created} .jsonl files.")
        return truncated + created


    @task(pool="default_pool")
    def transcribe_file(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        temp_path = TEMP_JSON_DIR / f"{file_id}.json"

        if temp_path.exists():
            return f"Skipped {file_id} (already transcribed)"

        res = requests.post(TRANSCRIBE_API_URL, json={"file_path": file_relative_path})
        res.raise_for_status()
        transcription = res.json()["transcription"].strip()

        with open(temp_path, "w") as f:
            json.dump({
                "id": file_relative_path,
                "transcription": transcription
            }, f)

        return f"Saved {file_id}"

    @task(pool="gpu_pool")
    def tokenise_file(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        temp_path = TEMP_JSON_DIR / f"{file_id}.json"

        if not temp_path.exists():
            raise FileNotFoundError(f"Missing transcription for {file_relative_path}")

        with open(temp_path) as f:
            partial = json.load(f)

        audio_np, _ = sf.read(AUDIO_ROOT / file_relative_path, dtype="float32")
        token_tensor = tokenise(audio_np)
        token_list = token_tensor.tolist()
        del token_tensor
        torch.cuda.empty_cache()

        folder_name = Path(file_relative_path).parts[0]
        output_file = RESULTS_DIR / f"{folder_name}.jsonl"
        lock_file = output_file.with_suffix(".lock")

        partial["token_array"] = token_list

        with FileLock(str(lock_file)):
            with open(output_file, "a") as out:
                out.write(json.dumps(partial) + "\n")

        del token_list
        return f"Written {file_id} to {folder_name}.jsonl"

    @task()
    def mark_tokenisation_complete():
        return "Tokenisation finished."

    @task(pool="default_pool")
    def cleanup_intermediate_jsons():
        deleted = 0
        for f in TEMP_JSON_DIR.glob("*.json"):
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}")
        return f"Deleted {deleted} temp JSONs"

    # === Direct DAG Execution (no list_audio_files) ===

    all_files = [
        str(p.relative_to(AUDIO_ROOT))
        for p in AUDIO_ROOT.rglob("*.flac")
    ]

    prepare = prepare_jsonl_files()

    transcribed = transcribe_file.expand(file_relative_path=all_files)
    transcribed.set_upstream(prepare)
    
    tokenised = tokenise_file.expand(file_relative_path=all_files)
    tokenised.set_upstream(transcribed)

    done = mark_tokenisation_complete()
    done.set_upstream(tokenised)

    cleanup_intermediate_jsons().set_upstream(done)
