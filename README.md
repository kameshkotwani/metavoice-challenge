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
Python 3.9+
uv (https://docs.astral.sh/uv/)
Apache Airflow 3.0
CUDA-enabled GPU
FastAPI Whisper server running locally at http://localhost:8000/transcribe


results/
├── transcriptions/
│   └── p225_001.json     # Contains transcription only
├── tokens/
│   └── p225_001.json     # Contains token_array only
├── p225.jsonl            # Combined results


# Run Instructions
Start your FastAPI server (for Whisper).

Start your Airflow webserver and scheduler.

Trigger the parallel_transcribe_tokenise DAG from the UI.

Notes

The mock tokenisation simulates GPU workload using tensor operations and artificial delay.

Output merging is safe with per-file locking.

.jsonl files are reset for every new run.

results/
├── transcriptions/
│   └── p225_001.json     # Contains transcription only
├── tokens/
│   └── p225_001.json     # Contains token_array only
├── p225.jsonl            # Combined results
