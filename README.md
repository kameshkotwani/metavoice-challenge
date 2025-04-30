# Parallel Audio Processing Pipeline

This project implements a scalable and horizontally parallel audio processing pipeline using **Apache Airflow**. It processes `.flac` audio files through two independent transformation steps:

1. **Transcription** using OpenAI Whisper (via a FastAPI server)
2. **Tokenisation** using a simulated GPU-heavy operation

The pipeline is optimised for **speed, safety, and concurrency**, capable of handling thousands of audio files in a robust production-grade environment.

---

## Features

- **Parallel execution** of transcription and tokenisation
- Tasks run independently (order does not matter)
- Results safely stored using per-folder `.jsonl` files
- Locking ensures data integrity during concurrent writes
- Fully scalable using Airflow’s task mapping and pools

---

## Directory Structure

project_root -> dags -> parallel_token_transcribe.py

project_root -> fast_api_server -> app.py


## How It Works

1. **list_files**: Lists all `.flac` audio files under `AUDIO_ROOT`.
2. **transcribe_file**: For each file, sends a request to the FastAPI Whisper server and stores the result in `results/transcriptions/`.
3. **tokenise_audio**: For each file, reads the audio and generates a dummy GPU tensor, saving to `results/tokens/`.
4. **merge_outputs**: Waits for both outputs (transcription and tokenisation), then combines them into a final `.jsonl` under `results/{folder}.jsonl`.

---

## Output Format

Each line in the final `.jsonl` looks like:

```json
{
  "id": "p225/p225_001_mic2.flac",
  "transcription": "Transcribed text goes here.",
  "token_array": [123, 456, 789, ...]
}
```

# Requirements

- Python 3.9+
- uv (https://docs.astral.sh/uv/)
- Apache Airflow 3.0
- CUDA-enabled GPU
- FastAPI Whisper server running locally at http://localhost:8000/transcribe

# Reason for choosing JSONL

- Line-by-line structure enables easy streaming — each audio file’s output is stored as a single line, making it easy to read or write incrementally without loading the full file into memory.

- Concurrency-safe — using file locking, multiple processes can append lines without corrupting the file, which is ideal for parallel pipelines.

- Well-supported by data tools — JSONL is natively supported by pandas, BigQuery, jq, Spark, and many modern analytics pipelines.

- Human-readable and version-controllable — each entry is a plain JSON object, making it inspectable and git-diff friendly.

- This format strikes a balance between performance, safety, and usability, especially in horizontally scalable Airflow DAGs where task outputs are merged independently.


# Results 

The results directory is not uploaded to github due to space limitations.
results
|
├── transcriptions/
│   └── p225_001.json     # Contains transcription only
├── tokens/
│   └── p225_001.json     # Contains token_array only
├── p225.jsonl            # Combined results


# Run Instructions

- install dependencies using uv ( uv sync)

- Start your FastAPI server (for Whisper).

- Start your Airflow webserver and scheduler. (airflow standalone)

- Trigger the parallel_transcribe_tokenise DAG from the UI.

## Optimisation of Transcription and Tokenisation Workflow in Airflow

### 1. Addressing Model Initialisation Bottleneck

- **Issue:** Each Airflow task independently loaded a Python instance of the transcription model (OpenAI Whisper), causing high memory usage and slow performance.
- **Solution:** A FastAPI server was developed to load the model once and serve concurrent transcription requests using multiple workers (e.g. `uvicorn --workers 2`).
- **Impact:**
  - Decoupled model execution from Airflow tasks, enabling independent horizontal scaling.
  - Reduced transcription latency from **8–10 seconds** per file to **~1 second** via API call.

### 2. Enabling Concurrent Transcription and Tokenisation

- **Issue:** Transcription and tokenisation were originally executed sequentially, resulting in underutilisation of system resources.
- **Solution:**
  - Split transcription and tokenisation into separate Airflow tasks, in line with atomic task principles (one operation per task).
  - Introduced an intermediate lock file mechanism to enable both processes to run in parallel as soon as partial data is available.
  - A final merge task combines transcription and token data by file ID and cleans up intermediate results.
- **Impact:**
  - Tokenisation starts independently and in parallel with transcription, improving throughput.
  - Both transcription and tokenisation tasks leverage the GPU, ensuring optimal hardware utilisation.
  - Scalable across CPU and GPU resources by increasing the number of Airflow workers.

### 3. Performance Benchmark (Tested on Local Machine - p225, 16GB RAM, 6 cores)

- **Sequential Execution:** Processing **462 files** took **40–45 minutes**.
- **Parallel Execution (FastAPI + atomic Airflow tasks):** Time reduced to **15–20 minutes**.
- **Scalability Outlook:** On production-grade infrastructure with API server and Airflow running independently, processing **500 files** is projected to complete in **4–5 minutes**, with horizontal scaling of workers and GPU nodes.
- The mock tokenisation simulates GPU workload using tensor operations and artificial delay.

Output merging is safe with per-file locking.

.jsonl files are reset for every new run.

results/


├── transcriptions/
│   └── p225_001.json     # Contains transcription only

├── tokens/

│   └── p225_001.json     # Contains token_array only

├── p225.jsonl            # Combined results


## Total time taken

The infrastructure setup and DAG operations were completed over a span of approximately 6 to 7 hours to fully accommodate the specified requirements.
