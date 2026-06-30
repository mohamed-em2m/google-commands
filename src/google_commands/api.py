"""FastAPI server for audio command recognition."""

import base64
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from google_commands.config import LABELS
from google_commands.pipeline import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Google Commands API",
    description="Real-time audio command recognition using ensemble deep learning.",
    version="1.0.0",
)

predictor = Predictor()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/labels")
async def get_labels():
    return {"labels": LABELS, "count": len(LABELS)}


@app.post("/predict", summary="Predict command from uploaded audio file")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        result = predictor.predict_from_file(tmp_path)
        return {"prediction": result, "filename": file.filename}
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/predict/b64", summary="Predict command from base64-encoded audio")
async def predict_b64(payload: dict):
    raw = payload.get("audio")
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Missing 'audio' field"})
    try:
        data = base64.b64decode(raw)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid base64"})
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = predictor.predict_from_file(tmp_path)
        return {"prediction": result}
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
