import base64
import io
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.inference_real import RetinaInference

# ─── App lifecycle ────────────────────────────────────────────────────────────

inference_engine: RetinaInference = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine
    print("Loading RetinaAI models...")
    inference_engine = RetinaInference()
    inference_engine.load_models()
    print("Models ready.")
    yield
    print("Shutting down.")


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RetinaAI API",
    description="Multi-disease retinal classification with anomaly detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 50
INPUT_SIZE = 256  # diffusion model input resolution


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode, convert to RGB, resize to INPUT_SIZE x INPUT_SIZE, normalize to [0,1]."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _apply_heatmap(gray: np.ndarray) -> np.ndarray:
    """Convert a (H,W) float32 [0,1] grayscale map to an RGBA heatmap.
    Low values → transparent, high values → red/yellow, fully opaque."""
    # Normalise to 0-1
    mn, mx = gray.min(), gray.max()
    if mx - mn > 1e-8:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray)

    # Boost contrast so subtle anomalies are visible
    gray = np.power(gray, 0.5)

    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Colour ramp: 0→blue, 0.25→cyan, 0.5→green, 0.75→yellow, 1→red
    t = gray
    rgba[..., 0] = np.clip(1.5 - np.abs(t - 1.0) * 4, 0, 1)    # R
    rgba[..., 1] = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)    # G
    rgba[..., 2] = np.clip(1.5 - np.abs(t - 0.0) * 4, 0, 1)    # B
    # Alpha: transparent at 0, fully opaque at 1
    rgba[..., 3] = np.clip(gray * 1.8, 0, 1)

    return (rgba * 255).astype(np.uint8)


def ndarray_to_base64_png(arr: np.ndarray, as_heatmap: bool = False) -> str:
    """Convert a float32 numpy array to a base64-encoded PNG string.
    If as_heatmap=True, applies a coloured RGBA heatmap to a 2-D array."""
    if arr.ndim == 2 and as_heatmap:
        rgba = _apply_heatmap(arr)
        img = Image.fromarray(rgba, mode="RGBA")
    elif arr.ndim == 2:
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-8:
            arr = (arr - mn) / (mx - mn)
        uint8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(uint8, mode="L")
    else:
        uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(uint8, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": inference_engine is not None and inference_engine.is_loaded,
        "input_size": INPUT_SIZE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate ──────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        image_array = preprocess_image(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = inference_engine.predict(image_array)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # ── Encode attention map → base64 PNG heatmap ─────────────────────────────
    attention_b64 = ndarray_to_base64_png(result["attention_map"], as_heatmap=True)

    # ── Build response ────────────────────────────────────────────────────────
    predictions = result["predictions"]  # dict of { disease: { probability, severity, description } }
    anomaly_score = float(result["anomaly_score"])

    # Primary diagnosis = highest probability disease
    primary = max(predictions, key=lambda d: predictions[d]["probability"])
    primary_prob = predictions[primary]["probability"]

    risk_level = (
        "High" if primary_prob > 0.7
        else "Moderate" if primary_prob > 0.4
        else "Low"
    )

    return JSONResponse({
        "scan_id": str(uuid.uuid4()),
        "inference_ms": elapsed_ms,
        "anomaly_score": anomaly_score,
        "predictions": predictions,
        "meta": {
            "primary_diagnosis": primary,
            "primary_probability": primary_prob,
            "risk_level": risk_level,
        },
        "attention_map_b64": attention_b64,   # base64 PNG, draw on top of the fundus image
    })
