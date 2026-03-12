## Running RetinaAI locally

### 1 - Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at http://localhost:8000/docs
Health check at http://localhost:8000/health

### 2 - Frontend (Vite)

```bash
cd Frontend
npm install
npm run dev
```

App at http://localhost:5173

---

### Plugging in your real models

1.  Drop weight files into  `backend/weights/`
2.  Open  `backend/models/inference.py`
3.  Follow every  `# TODO ▸`  comment — each one is one model stage
4.  Uncomment the relevant framework line in  `requirements.txt`  (torch / tensorflow)
5.  Restart uvicorn — the frontend needs no changes

### API contract (POST /predict)

**Request:** multipart/form-data with field `file` (JPEG or PNG)

**Response JSON:**
```json
{
  "scan_id": "uuid",
  "inference_ms": 450,
  "anomaly_score": 0.78,
  "predictions": {
    "diabetic_retinopathy": { "probability": 0.85, "severity": "Moderate DR", "description": "..." },
    "glaucoma":              { "probability": 0.12, "severity": "No Glaucoma",  "description": "..." },
    "pathologic_myopia":     { "probability": 0.23, "severity": "No Pathologic Myopia", "description": "..." }
  },
  "meta": {
    "primary_diagnosis": "diabetic_retinopathy",
    "primary_probability": 0.85,
    "risk_level": "High"
  },
  "attention_map_b64": "<base64 PNG, 256x256 grayscale>"
}
```
