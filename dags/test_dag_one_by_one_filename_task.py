from airflow import DAG
from airflow.decorators import task
from datetime import timedelta
from pathlib import Path
import logging
import json
import whisper
import torch
import numpy as np
import soundfile as sf
import requests
from filelock import FileLock
from functools import lru_cache
from time import time
from config import DATA_DIR,RAW_DATA_DIR,TEST_DATA_DIR,JSON_DIR,RESULTS_DIR

# Config paths
AUDIO_ROOT = TEST_DATA_DIR

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


# === DAG DEFINITION ===
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="audio_transcribe_tokenise_dynamic_filename_task_id",
    default_args=default_args,
    description="Transcribe and tokenise each audio file in parallel",
    catchup=False,
    tags=["audio", "whisper", "tokenise", "dynamic"],
    max_active_tasks=4
) as dag:

    @task(retries=2, retry_delay=timedelta(minutes=1), pool="default_pool")  # adjust pool name if needed
    def list_audio_files():
        """List all .flac files recursively under AUDIO_ROOT"""
        return [
            str(p.relative_to(AUDIO_ROOT))
            for folder in AUDIO_ROOT.iterdir() if folder.is_dir()
            for p in folder.glob("*.flac")
        ]


    # === Airflow Task ===
    @task(retries=2, retry_delay=timedelta(minutes=1), pool="gpu_pool")  # adjust pool name if needed
    def process_file(file_relative_path: str):
        log = logging.getLogger(__name__)
        file_path = AUDIO_ROOT / file_relative_path

        try:
            log.info(f"Processing: {file_relative_path}")
            start_time = time()

            # === Step 1: Transcribe via FastAPI
            response = requests.post(TRANSCRIBE_API_URL, json={"file_path": file_relative_path})
            response.raise_for_status()
            transcription = response.json()["transcription"].strip()
            log.info(f"Transcription done: {transcription[:50]}...")

            # === Step 2: Tokenise
            audio_np, _ = sf.read(file_path, dtype="float32")
            token_tensor = tokenise(audio_np)
            token_list = token_tensor.tolist()
            log.info(f"Tokenisation done: {len(token_list)} tokens")

            # === Step 3: Write to .jsonl safely
            output_record = {
                "id": file_relative_path,
                "transcription": transcription,
                "token_array": token_list,
            }

            with FileLock(str(LOCK_FILE)):
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(output_record) + "\n")

            log.info(f"File processed: {file_relative_path} in {round((time() - start_time) * 1000)} ms")
            
            
        except Exception as e:
            log.error(f"Error: {file_relative_path} → {e}")
            raise


    # === DAG EXECUTION FLOW ===
    audio_files = list_audio_files()
    process_file.expand(file_relative_path=audio_files)
