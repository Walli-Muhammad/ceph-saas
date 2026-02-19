"""
FastAPI application for Cephalometric Analysis API

Endpoints:
  GET  /health   → Health check + model status
  POST /analyze  → Upload X-ray → return landmarks + SNA/SNB/ANB angles

Start server:
  backend\\venv\\Scripts\\uvicorn backend.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

import sys
sys.path.append(str(Path(__file__).parent))

from core.ml.inference import CephPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load model once at startup
# ─────────────────────────────────────────────────────────────────────────────
predictor: CephPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; clean up on shutdown."""
    global predictor
    logger.info("Loading CephPredictor model…")
    try:
        predictor = CephPredictor()
        logger.info("✓ Model ready.")
    except FileNotFoundError as e:
        logger.error(f"✗ Model not found: {e}")
        predictor = None

    yield  # ← server is running here

    logger.info("Server shutdown — releasing resources.")
    predictor = None


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ceph-SaaS Inference API",
    description=(
        "Deep-learning cephalometric analysis: "
        "upload a lateral skull X-ray and receive "
        "29 landmark coordinates plus SNA / SNB / ANB angles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development (tighten before production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health_check():
    """Return API health and model load status."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/analyze", tags=["Inference"])
async def analyze(file: UploadFile = File(...)):
    """
    Run cephalometric analysis on an uploaded X-ray image.

    - **file**: Lateral cephalogram (JPG / PNG / BMP)

    Returns:
    - **landmarks**: 29 named coordinates in original-image pixel space
    - **analysis**: SNA, SNB, ANB angles in degrees
    - **message**: "Success"
    """
    # Guard: model must be loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Ensure best_model.pth exists in backend/checkpoints/ "
                "and restart the server."
            ),
        )

    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPG, PNG, or BMP.",
        )

    # Read bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run inference
    try:
        result = predictor.predict(image_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "message": "Success",
        "landmarks": result["landmarks"],
        "analysis": result["analysis"],
    }


@app.post(
    "/analyze/visualize",
    tags=["Inference"],
    response_class=StreamingResponse,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def analyze_visualize(file: UploadFile = File(...)):
    """
    Run cephalometric analysis and return the **annotated X-ray image**.

    - **file**: Lateral cephalogram (JPG / PNG / BMP)

    Returns a JPEG image with:
    - 🔴 Red dot at each of the 29 predicted landmark positions
    - 📋 SNA / SNB / ANB angle values in the top-left corner
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPG, PNG, or BMP.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        jpeg_bytes = predictor.visualize(image_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Visualize error")
        raise HTTPException(status_code=500, detail=f"Visualisation failed: {e}")

    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=analysis.jpg"},
    )
