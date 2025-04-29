from fastapi import FastAPI, Body, HTTPException
import whisper
import os
from config import TEST_DATA_DIR
import soundfile as sf
from threading import Semaphore
import logging

app = FastAPI()

# Load Whisper model once (on CPU or "cuda")
model = whisper.load_model("tiny",in_memory=True)

# Limit concurrency (e.g., 20 concurrent transcriptions)
semaphore = Semaphore(20)

# Base directory for audio files
AUDIO_ROOT = TEST_DATA_DIR 

logger = logging.getLogger("uvicorn.error")

@app.post("/transcribe")
async def transcribe(file_path: str = Body(..., embed=True)):
    full_path = os.path.join(AUDIO_ROOT, file_path)

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Read and validate audio
    try:
        audio_np, sr = sf.read(full_path)
        if audio_np.shape[0] == 0 or len(audio_np) < 1600:
            raise HTTPException(status_code=400, detail="Audio too short or empty")
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        raise HTTPException(status_code=400, detail="Failed to load audio file")

    # Acquire concurrency lock
    if not semaphore.acquire(timeout=10):
        raise HTTPException(status_code=429, detail="Too many concurrent requests")

    try:
        result = model.transcribe(full_path)
        transcription = result["text"].strip()
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        raise HTTPException(status_code=500, detail="Whisper transcription failed")
    finally:
        semaphore.release()

# Global error handler (optional)
@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    raise HTTPException(status_code=500, detail="Internal server error")

