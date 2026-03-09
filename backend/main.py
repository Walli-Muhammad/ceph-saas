import base64
import json
import logging
import math as _math
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image as _PILImage

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
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
predictor: CephPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading CephPredictor model...")
    try:
        predictor = CephPredictor()
        logger.info("Model ready.")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        predictor = None
    yield
    logger.info("Server shutdown.")
    predictor = None


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ceph-SaaS Inference API",
    description="Deep-learning cephalometric analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
async def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze  (landmarks + analysis JSON, no image)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze", tags=["Inference"])
async def analyze(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        result = predictor.predict(image_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {"message": "Success", "landmarks": result["landmarks"], "analysis": result["analysis"]}


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/visualize  (returns JPEG stream)
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/analyze/visualize",
    tags=["Inference"],
    response_class=StreamingResponse,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def analyze_visualize(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
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


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/full  (base64 image + diagnostics table + landmarks)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze/full", tags=["Inference"])
async def analyze_full(
    file:     UploadFile      = File(...),
    calib_x1: Optional[float] = Form(None),
    calib_y1: Optional[float] = Form(None),
    calib_x2: Optional[float] = Form(None),
    calib_y2: Optional[float] = Form(None),
    calib_mm: Optional[float] = Form(None),
):
    """
    1. Reads an image (JPG/PNG).
    2. Runs U-Net inference to find 29 marks.
    3. Calculates diagnostics (SNA, SNB, mm lengths).
    4. Overlays analysis on the image.
    5. Returns JSON table, base64 image slice, pixel_size, and landmarks dict.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    from core.ml.inference import lookup_pixel_size
    csv_pixel_size = lookup_pixel_size(file.filename)
    
    active_pixel_size = csv_pixel_size if csv_pixel_size is not None else 0.1
    calibration_status = "none"

    if all(v is not None for v in [calib_x1, calib_y1, calib_x2, calib_y2, calib_mm]) and calib_mm > 0:
        try:
            img = _PILImage.open(io.BytesIO(image_bytes))
            native_w, native_h = img.size
            px1 = calib_x1 * native_w; py1 = calib_y1 * native_h
            px2 = calib_x2 * native_w; py2 = calib_y2 * native_h
            native_dist = _math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
            
            if native_dist > 0:
                ruler_pixel_size = calib_mm / native_dist
                logger.info(f"=== CALIBRATION VALIDATION ===")
                logger.info(f"Ruler MM/PX: {ruler_pixel_size:.6f}")
                logger.info(f"CSV MM/PX  : {csv_pixel_size}")
                
                if csv_pixel_size is None:
                    # No CSV ground truth exists -> trust the ruler unconditionally
                    active_pixel_size = ruler_pixel_size
                    calibration_status = "accepted_no_csv"
                    logger.info("Status: accepted_no_csv (No CSV mapping found)")
                else:
                    ratio = ruler_pixel_size / csv_pixel_size
                    logger.info(f"Ratio: {ratio:.3f}")
                    if 0.85 <= ratio <= 1.15:
                        active_pixel_size = ruler_pixel_size
                        calibration_status = "accepted"
                        logger.info("Status: accepted (within 15% tolerance)")
                    else:
                        active_pixel_size = csv_pixel_size
                        calibration_status = "rejected"
                        logger.warning("Status: rejected (out of tolerance, using CSV fallback)")
                        
        except Exception as exc:
            logger.warning(f"Calibration math failed ({exc}) — using CSV default.")

    try:
        result     = predictor.predict(image_bytes, filename=file.filename, override_pixel_size=active_pixel_size)
        jpeg_bytes = predictor.visualize(image_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("analyze/full error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    landmarks_out = {
        name: {"x": float(xy[0]), "y": float(xy[1])}
        for name, xy in result["landmarks"].items()
    }

    from core.ml.llm_service import generate_summary
    clinical_summary = generate_summary(result["diagnostics_table"])

    return {
        "message":            "Success",
        "diagnostics_table":  result["diagnostics_table"],
        "clinical_summary":   clinical_summary,
        "image_base64":       base64.b64encode(jpeg_bytes).decode("utf-8"),
        "pixel_size":         active_pixel_size,
        "calibration_status": calibration_status,
        "landmarks":          landmarks_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/adjust  (re-render with corrected landmarks, no PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze/adjust", tags=["Inference"])
async def analyze_adjust(
    file:           UploadFile = File(...),
    landmarks_json: str        = Form(...),
):
    """
    Re-render the cephalometric image using manually corrected landmark positions.
    Skips PyTorch inference -- ideal for interactive drag-and-correct flows.

    - file: The original X-ray (used for rendering, not re-analysed)
    - landmarks_json: JSON string {name: {x, y}} in native image pixels
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        landmarks: dict = json.loads(landmarks_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid landmarks_json: {e}")
    
    from core.ml.inference import lookup_pixel_size
    pixel_size = lookup_pixel_size(file.filename)

    try:
        jpeg_bytes, diagnostics_table = predictor.visualize_from_landmarks(
            image_bytes, landmarks, pixel_size=pixel_size
        )
    except Exception as e:
        logger.exception("analyze/adjust error")
        raise HTTPException(status_code=500, detail=f"Adjust render failed: {e}")

    landmarks_out = {}
    for name, val in landmarks.items():
        if isinstance(val, dict):
            landmarks_out[name] = {"x": float(val["x"]), "y": float(val["y"])}
        else:
            landmarks_out[name] = {"x": float(val[0]), "y": float(val[1])}

    from core.ml.llm_service import generate_summary
    clinical_summary = generate_summary(diagnostics_table)

    return {
        "message":           "Success",
        "diagnostics_table": diagnostics_table,
        "clinical_summary":  clinical_summary,
        "image_base64":      base64.b64encode(jpeg_bytes).decode("utf-8"),
        "pixel_size":        pixel_size,
        "landmarks":         landmarks_out,
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/chat
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    diagnostics: list
    question: str

@app.post("/analyze/chat", tags=["Inference"])
async def analyze_chat(req: ChatRequest):
    """
    Ask the AI Assistant a specific question about the patient's diagnostic results.
    """
    from core.ml.llm_service import ask_question
    try:
        answer = ask_question(req.diagnostics, req.question)
        return {"answer": answer}
    except Exception as e:
        logger.exception("analyze/chat error")
        raise HTTPException(status_code=500, detail=f"AI Chat failed: {e}")
