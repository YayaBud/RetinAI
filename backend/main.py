import base64
import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
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
    """Convert a (H,W) float32 [0,1] grayscale map to an RGBA heatmap
    using matplotlib's jet colormap — matching the reference anomaly maps.
    Low values → transparent, high values → opaque red/yellow."""
    
    gray = np.clip(gray, 0, 1)

    try:
        import matplotlib.cm as cm
        colored = cm.jet(gray)  # (H, W, 4) RGBA float in [0,1]
        rgba = colored.astype(np.float32)
    except ImportError:
        # Fallback: manual jet-ish colormap
        h, w = gray.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        t = gray
        rgba[..., 0] = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)   # R
        rgba[..., 1] = np.clip(1.5 - np.abs(t - 0.5)  * 4, 0, 1)   # G
        rgba[..., 2] = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)   # B
        rgba[..., 3] = 1.0

    # Alpha: transparent where anomaly is low, opaque where high
    alpha = np.clip((gray - 0.05) / 0.35, 0, 1)  # ramp from 0.05→0.40
    alpha = alpha ** 0.5  # open up mid-values more
    rgba[..., 3] = alpha

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


# ─── AI Chat (Ollama) ─────────────────────────────────────────────────────────

OPHTHALMOLOGY_SYSTEM_PROMPT = (
    "You are RetinAI Assistant, a specialized ophthalmology AI assistant "
    "integrated with a retinal scan analysis system.\n\n"
    "Your expertise covers:\n"
    "- Diabetic Retinopathy (DR): stages from mild NPDR to PDR, microaneurysms, "
    "hemorrhages, hard/soft exudates, neovascularization, macular edema\n"
    "- Glaucoma: open-angle, angle-closure, cup-to-disc ratio, nerve fiber layer, "
    "IOP management\n"
    "- Pathologic Myopia: posterior staphyloma, lacquer cracks, Fuchs spot, myopic CNV\n"
    "- General ophthalmology: AMD, retinal detachment, vascular occlusions, "
    "optic neuropathies\n\n"
    "When scan results are provided, analyze them thoroughly: explain probability "
    "scores, suggest follow-up tests, discuss treatment options, and note urgency.\n\n"
    "Guidelines:\n"
    "- Use proper medical terminology\n"
    "- Always note this is AI-assisted analysis, not a definitive diagnosis\n"
    "- Recommend professional clinical evaluation for concerning findings\n"
    "- Be concise but thorough"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    system_prompt = OPHTHALMOLOGY_SYSTEM_PROMPT
    if request.scan_context:
        system_prompt += f"\n\nCurrent patient scan results:\n{request.scan_context}"

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in request.messages],
    ]

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={"model": "llama3.2:1b", "messages": messages, "stream": False},
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Ollama returned an error")

        data = resp.json()
        return {"response": data["message"]["content"]}
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )


@app.get("/ollama-status")
async def ollama_status():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "models": models}
    except Exception:
        return {"status": "unavailable", "models": []}
