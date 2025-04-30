from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
from pathlib import Path
import json
import torch
import numpy as np
import soundfile as sf
import requests
from filelock import FileLock
import time
from config import TEST_DATA_DIR, RESULTS_DIR

# === CONFIG ===
AUDIO_ROOT = TEST_DATA_DIR
TEMP_JSON_DIR = RESULTS_DIR / "transcriptions"
TOKEN_JSON_DIR = RESULTS_DIR / "tokens"
RESULTS_DIR = RESULTS_DIR
TRANSCRIBE_API_URL = "http://localhost:8000/transcribe"

# === SETUP ===
for d in [TEMP_JSON_DIR, TOKEN_JSON_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === TOKENISE MOCK ===
def tokenise(audio_np_array: np.ndarray) -> torch.Tensor:
    """
    Function to tokenise an audio file represented as a NumPy array.

    Args:
    - audio_np_array (np.ndarray): The audio file as a NumPy array.

    Returns:
    - torch.Tensor: A random 1D tensor with dtype int16 and variable length in range (20, 1000).
    """
    if not isinstance(audio_np_array, np.ndarray):
        raise ValueError("Input should be a NumPy array")

    del audio_np_array  # unused

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. This operation requires a CUDA-capable GPU.")

    start_time = time.time()

    while True:
        tensor_length = np.random.randint(20, 1001)  # 1001 is exclusive
        result_tensor = torch.randint(low=-32768, high=32767, size=(tensor_length,), dtype=torch.int16, device='cuda')

        # Perform a dummy matrix multiplication to engage the GPU
        a = torch.rand(5000, 5000, device='cuda')
        b = torch.rand(5000, 5000, device='cuda')
        _ = torch.matmul(a, b)  # Result is not used, just to simulate work

        elapsed_time_ms = (time.time() - start_time) * 1000
        if elapsed_time_ms >= 200:
            print(f'elapsed_time_ms: {elapsed_time_ms}')
            break

    return result_tensor

# === DAG ===
default_args = {"retries": 3, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="parallel_transcribe_tokenise",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_tasks=10,
    tags=["parallel", "whisper", "tokenise","gpu_pool"],
) as dag:

    @task()
    def list_files():
        return [str(p.relative_to(AUDIO_ROOT)) for p in AUDIO_ROOT.rglob("*.flac")]

    @task(pool='default_pool')
    def transcribe_file(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        out_path = TEMP_JSON_DIR / f"{file_id}.json"
        if out_path.exists(): return file_id
        res = requests.post(TRANSCRIBE_API_URL, json={"file_path": file_relative_path})
        res.raise_for_status()
        with open(out_path, "w") as f:
            json.dump({"id": file_relative_path, "transcription": res.json()["transcription"]}, f)
        return file_id

    @task(pool='gpu_pool')
    def tokenise_audio(file_relative_path: str):
        file_id = Path(file_relative_path).stem
        out_path = TOKEN_JSON_DIR / f"{file_id}.json"
        if out_path.exists(): return file_id
        audio_np, _ = sf.read(AUDIO_ROOT / file_relative_path, dtype="float32")
        token_tensor = tokenise(audio_np)
        with open(out_path, "w") as f:
            json.dump({"id": file_relative_path, "token_array": token_tensor.tolist()}, f)
        del token_tensor
        del audio_np
        torch.cuda.empty_cache()
        return file_id

    @task(pool="default_pool")
    def merge_outputs(file_id: str):
        transcript_file = TEMP_JSON_DIR / f"{file_id}.json"
        token_file = TOKEN_JSON_DIR / f"{file_id}.json"
        if not transcript_file.exists() or not token_file.exists():
            raise FileNotFoundError("Missing intermediate data")
        with open(transcript_file) as f:
            transcript = json.load(f)
        with open(token_file) as f:
            token = json.load(f)
        merged = transcript
        merged["token_array"] = token["token_array"]
        folder = Path(transcript["id"]).parts[0]
        out_file = RESULTS_DIR / f"{folder}.jsonl"
        lock = FileLock(out_file.with_suffix(".lock"))
        with lock:
            with open(out_file, "a") as f:
                f.write(json.dumps(merged) + "\n")
        return f"Merged {file_id}"

    files = list_files()
    trans_ids = transcribe_file.expand(file_relative_path=files)
    token_ids = tokenise_audio.expand(file_relative_path=files)
    merge_outputs.expand(file_id=files).set_upstream([trans_ids, token_ids])
