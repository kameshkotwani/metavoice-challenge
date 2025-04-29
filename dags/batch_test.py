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
BATCH_SIZE=5
# === TOKENISER FUNCTION ===
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

# === MODEL LOADER ===
@lru_cache(maxsize=1)
def get_whisper_model():
    return whisper.load_model(MODEL_NAME, device=DEVICE)

# === DAG DEFINITION ===
default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="audio_batch_transcribe_tokenise",
    default_args=default_args,
    description="Batch transcribe and tokenise audio files",
    catchup=False,
    tags=["whisper", "batch", "gpu"],
    max_active_tasks=4,  # adjust for your GPU
) as dag:

    @task()
    def list_file_batches():
        """Group .flac files into batches of size BATCH_SIZE"""
        all_files = [
            str(p.relative_to(AUDIO_ROOT))
            for folder in AUDIO_ROOT.iterdir() if folder.is_dir()
            for p in folder.glob("*.flac")
        ]
        return [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]

    @task()
    def process_file_batch(file_paths: list[str]):
        log = logging.getLogger(__name__)
        model = get_whisper_model()

        for rel_path in file_paths:
            file_path = AUDIO_ROOT / rel_path
            try:
                log.info(f"▶️ Processing: {rel_path}")
                start_time = time()

                # Transcribe
                result = model.transcribe(str(file_path))
                transcription = result["text"].strip()

                # Tokenise
                audio_np, _ = sf.read(file_path, dtype="float32")
                token_tensor = tokenise(audio_np)
                token_list = token_tensor.tolist()

                output_record = {
                    "id": rel_path,
                    "transcription": transcription,
                    "token_array": token_list,
                }

                with FileLock(str(LOCK_FILE)):
                    with open(OUTPUT_FILE, "w") as f:
                        f.write(json.dumps(output_record) + "\n")

                log.info(f"✅ Done: {rel_path} ({len(token_list)} tokens in {round((time() - start_time) * 1000)} ms)")

            except Exception as e:
                log.error(f"❌ Failed: {rel_path} → {e}")
                raise

    batches = list_file_batches()
    process_file_batch.expand(file_paths=batches)