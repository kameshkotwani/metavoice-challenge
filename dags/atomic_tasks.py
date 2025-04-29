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
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="audio_pipeline_split_atomic",
    default_args=default_args,
    catchup=False,
    max_active_tasks=6,
    tags=["atomic", "whisper", "tokenise","fastapi-server","gpu_pool,cpu_pool"]
) as dag:

    @task()
    def list_audio_files():
        return [
            str(p.relative_to(AUDIO_ROOT))
            for folder in AUDIO_ROOT.iterdir() if folder.is_dir()
            for p in folder.glob("*.flac")
        ]

    @task(pool="default_pool")  # use CPU pool for transcription
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

        partial["token_array"] = token_list

        with FileLock(str(LOCK_FILE)):
            with open(OUTPUT_FILE, "a") as out:
                out.write(json.dumps(partial) + "\n")

        return f"Written {file_id}"

    audio_files = list_audio_files()
    transcribed = transcribe_file.expand(file_relative_path=audio_files)
    tokenise_file.expand(file_relative_path=audio_files).set_upstream(transcribed)