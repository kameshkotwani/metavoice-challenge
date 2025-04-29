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
from filelock import FileLock
from functools import lru_cache
from time import time
from config import DATA_DIR,RAW_DATA_DIR,TEST_DATA_DIR,JSON_DIR,RESULTS_DIR

# Config paths
AUDIO_ROOT = TEST_DATA_DIR

OUTPUT_FILE = RESULTS_DIR / "all_results.jsonl"

LOCK_FILE = OUTPUT_FILE.with_suffix(".lock")

MODEL_NAME = "tiny"

DEVICE = "cuda"  # or "cpu"
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

# === CACHED MODEL LOADER ===
@lru_cache(maxsize=1)
def get_whisper_model():
    return whisper.load_model(MODEL_NAME, device=DEVICE)

# === DAG DEFINITION ===
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="audio_transcribe_tokenise_dynamic",
    default_args=default_args,
    description="Transcribe and tokenise each audio file in parallel",
    catchup=False,
    tags=["audio", "whisper", "tokenise", "dynamic"],
    max_active_tasks=4
) as dag:

    @task()
    def list_audio_files():
        """List all .flac files recursively under AUDIO_ROOT"""
        return [
            str(p.relative_to(AUDIO_ROOT))
            for folder in AUDIO_ROOT.iterdir() if folder.is_dir()
            for p in folder.glob("*.flac")
        ]

    @task(pool="gpu_pool", retries=2, retry_delay=timedelta(minutes=1))
    def process_file(file_relative_path: str):
        """Process a single file: transcribe + tokenise + save output"""
        log = logging.getLogger(__name__)
        file_path = AUDIO_ROOT / file_relative_path

        try:
            log.info(f"Starting: {file_relative_path}")
            start_time = time()

            # Load whisper model
            model = get_whisper_model()

            # Transcription
            result = model.transcribe(str(file_path))
            transcription = result["text"].strip()
            log.info(f"Transcribed: \"{transcription[:50]}...\"")

            # Load audio and tokenise
            audio_np, _ = sf.read(file_path, dtype="float32")
            token_tensor = tokenise(audio_np)
            token_list = token_tensor.tolist()
            log.info(f"Tokenised: {len(token_list)} tokens")

            # Format output
            output_record = {
                "id": file_relative_path,
                "transcription": transcription,
                "token_array": token_list,
            }

            # Write safely with file lock
            with FileLock(str(LOCK_FILE)):
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(output_record) + "\n")

            log.info(f"Written: {file_relative_path}")
            log.info(f"Time taken: {round((time() - start_time) * 1000)} ms")

        except Exception as e:
            log.error(f"Error in {file_relative_path}: {e}")
            raise  # Force Airflow to retry


    # === DAG EXECUTION FLOW ===
    audio_files = list_audio_files()
    process_file.expand(file_relative_path=audio_files)
