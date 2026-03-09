"""
Inference Engine for Cephalometric Landmark Detection

Provides CephPredictor class that:
  1. Loads the trained U-Net from checkpoints/best_model.pth
  2. Preprocesses raw image bytes (grayscale, resize 512x512, normalize)
  3. Runs inference to get 29 heatmaps
  4. Extracts (x, y) peak from each heatmap
  5. Scales coordinates back to original image dimensions
  6. Calculates clinical angles: SNA, SNB, ANB
"""

import csv
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
try:
    from scipy.interpolate import PchipInterpolator
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    logging.getLogger(__name__).warning(
        "scipy not found — spline curves will fall back to straight lines. "
        "Run: pip install scipy"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.ml.model import UNet
from core.ml.diagnostics import DiagnosticsEngine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Landmark index mapping  (order matches the annotation JSON as seen in training)
# ─────────────────────────────────────────────────────────────────────────────
LANDMARK_NAMES: List[str] = [
    "A-point",           # 0
    "ANS",               # 1
    "B-point",           # 2
    "Menton",            # 3
    "Nasion",            # 4
    "Orbitale",          # 5
    "Pogonion",          # 6
    "PNS",               # 7
    "Pronasale",         # 8
    "Ramus",             # 9
    "Sella",             # 10
    "Articulare",        # 11
    "Condylion",         # 12
    "Gnathion",          # 13
    "Gonion",            # 14
    "Porion",            # 15
    "Lower 2nd PM Cusp", # 16
    "Lower Incisor Tip", # 17
    "Lower Molar Cusp",  # 18
    "Upper 2nd PM Cusp", # 19
    "Upper Incisor Apex",# 20
    "Upper Incisor Tip", # 21
    "Upper Molar Cusp",  # 22
    "Lower Incisor Apex",# 23
    "Labrale Inferius",  # 24
    "Labrale Superius",  # 25
    "Soft Tissue Nasion",# 26
    "Soft Tissue Pog",   # 27
    "Subnasale",         # 28
]

# Indices for angle and distance calculations
IDX_A         = 0   # A-point
IDX_B         = 2   # B-point
IDX_NASION    = 4   # Nasion (N)
IDX_SELLA     = 10  # Sella (S)
IDX_CONDYLION = 12  # Condylion (Co)
IDX_GNATHION  = 13  # Gnathion (Gn)

TARGET_SIZE       = 512
DEFAULT_PIXEL_SIZE = 0.1  # mm/px fallback

# ─────────────────────────────────────────────────────────────────────────────
# CSV pixel-size lookup
# ─────────────────────────────────────────────────────────────────────────────

_PIXEL_SIZE_MAP: Optional[Dict[str, float]] = None
_CSV_PATH = (
    Path(__file__).parent.parent.parent.parent  # project root
    / "data" / "Aariz" / "cephalogram_machine_mappings.csv"
)


def _get_pixel_size_map() -> Dict[str, float]:
    """Load and cache the cephalogram_id → pixel_size mapping from the CSV."""
    global _PIXEL_SIZE_MAP
    if _PIXEL_SIZE_MAP is not None:
        return _PIXEL_SIZE_MAP

    mapping: Dict[str, float] = {}
    if _CSV_PATH.exists():
        with open(_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cephalogram_id", "").strip()
                try:
                    px = float(row.get("pixel_size", DEFAULT_PIXEL_SIZE))
                except ValueError:
                    px = DEFAULT_PIXEL_SIZE
                if cid:
                    mapping[cid] = px
        logger.info(f"Loaded pixel_size for {len(mapping)} cephalograms from CSV.")
    else:
        logger.warning(f"CSV not found at {_CSV_PATH}; using default pixel_size {DEFAULT_PIXEL_SIZE} mm/px.")

    _PIXEL_SIZE_MAP = mapping
    return _PIXEL_SIZE_MAP


def lookup_pixel_size(filename: Optional[str]) -> float:
    """
    Derive pixel_size (mm/px) from the uploaded filename.

    The filename is expected to be the cephalogram_id with an extension,
    e.g. "cks2ip8fp29yl0yuf6ry9266i.png". The extension is stripped before
    looking up in the CSV. Falls back to DEFAULT_PIXEL_SIZE if not found.
    """
    if not filename:
        return DEFAULT_PIXEL_SIZE
    stem = Path(filename).stem          # strip extension
    mapping = _get_pixel_size_map()
    ps = mapping.get(stem, DEFAULT_PIXEL_SIZE)
    if ps == DEFAULT_PIXEL_SIZE and stem not in mapping:
        logger.debug(f"'{stem}' not found in CSV; using default {DEFAULT_PIXEL_SIZE} mm/px.")
    return ps

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _angle_at_vertex(
    p1: Tuple[float, float],
    vertex: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """
    Compute the angle (in degrees) formed at `vertex` between rays
    vertex→p1 and vertex→p2, using atan2 for robustness.

    Positive = counter-clockwise from v1 to v2.
    For cephalometrics we return the absolute angle.
    """
    v1x = p1[0] - vertex[0]
    v1y = p1[1] - vertex[1]
    v2x = p2[0] - vertex[0]
    v2y = p2[1] - vertex[1]

    dot   = v1x * v2x + v1y * v2y
    cross = v1x * v2y - v1y * v2x   # z-component of cross product

    angle_rad = math.atan2(abs(cross), dot)
    return math.degrees(angle_rad)


# ─────────────────────────────────────────────────────────────────────────────
# CephPredictor
# ─────────────────────────────────────────────────────────────────────────────

class CephPredictor:
    """
    End-to-end cephalometric inference.

    Usage:
        predictor = CephPredictor()
        result = predictor.predict(image_bytes)
    """

    def __init__(self, checkpoint_path: Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CephPredictor using device: {self.device}")

        # Resolve checkpoint path
        if checkpoint_path is None:
            # Path(__file__) = backend/core/ml/inference.py
            # .parent.parent.parent  = backend/
            checkpoint_path = (
                Path(__file__).parent.parent.parent
                / "checkpoints" / "best_model.pth"
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}\n"
                "Please train the model first: python backend/core/ml/train.py"
            )

        # Build and load model
        self.model = UNet(in_channels=1, out_channels=29, bilinear=True).to(self.device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("best_val_loss", "?")
        logger.info(f"Loaded model from epoch {epoch} (val_loss={val_loss})")

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, image_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert raw image bytes → normalised 512×512 tensor.

        Returns:
            tensor : (1, 1, 512, 512) float32 on self.device
            orig_size : (width, height) of the original image
        """
        # Decode bytes to numpy BGR array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image. Ensure the file is a valid image.")

        orig_h, orig_w = img_bgr.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalise [0, 255] → [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch + channel dims → (1, 1, 512, 512)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)

        return tensor, (orig_w, orig_h)

    # ── Postprocessing ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_peaks(heatmaps_np: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find (x, y) of the brightest pixel in each of the 29 heatmaps.

        heatmaps_np: (29, 512, 512)
        Returns list of (x, y) in 512×512 coordinate space.
        """
        peaks = []
        for hm in heatmaps_np:
            flat_idx = int(np.argmax(hm))
            y, x = divmod(flat_idx, TARGET_SIZE)
            peaks.append((x, y))
        return peaks

    @staticmethod
    def _scale_to_original(
        peaks_512: List[Tuple[int, int]],
        orig_size: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """Scale 512-space coords back to original image space."""
        orig_w, orig_h = orig_size
        scale_x = orig_w / TARGET_SIZE
        scale_y = orig_h / TARGET_SIZE
        return [(x * scale_x, y * scale_y) for x, y in peaks_512]

    # ── Clinical analysis ────────────────────────────────────────────────────

    @staticmethod
    def calculate_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        pixel_size: float,
    ) -> float:
        """
        Euclidean distance between two points (in original-image pixel space),
        converted to millimetres using the machine-specific pixel_size.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_px = math.sqrt(dx * dx + dy * dy)
        return round(dist_px * pixel_size, 2)

    @staticmethod
    def _compute_analysis(
        landmarks_orig: List[Tuple[float, float]],
        pixel_size: float,
    ) -> Dict[str, float]:
        """
        Compute SNA/SNB/ANB angles and linear bone measurements.

        Angles:   SNA, SNB, ANB
        Lengths:  Mandibular Length (Co-Gn), Maxillary Length (Co-A)  [mm]
        """
        S  = landmarks_orig[IDX_SELLA]
        N  = landmarks_orig[IDX_NASION]
        A  = landmarks_orig[IDX_A]
        B  = landmarks_orig[IDX_B]
        Co = landmarks_orig[IDX_CONDYLION]
        Gn = landmarks_orig[IDX_GNATHION]

        SNA = _angle_at_vertex(S, N, A)
        SNB = _angle_at_vertex(S, N, B)
        ANB = round(SNA - SNB, 2)

        dx = lambda p1, p2: math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        mandibular_mm = round(dx(Co, Gn) * pixel_size, 2)
        maxillary_mm  = round(dx(Co, A)  * pixel_size, 2)

        return {
            "SNA": round(SNA, 2),
            "SNB": round(SNB, 2),
            "ANB": ANB,
            "Mandibular Length (mm)": mandibular_mm,
            "Maxillary Length (mm)": maxillary_mm,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        override_pixel_size: Optional[float] = None,
    ) -> Dict:
        """
        Full inference pipeline.

        Args:
            image_bytes: Raw bytes of a cephalogram image (JPG / PNG / BMP).
            filename: Original upload filename (used to look up pixel_size).
            override_pixel_size: If provided by the client (calibrated ruler),
                overrides the CSV-derived pixel size for all mm calculations.
                Angles are unaffected (they're purely geometric).
        """
        # 1. Preprocess
        tensor, orig_size = self._preprocess(image_bytes)

        # 2. Inference
        with torch.no_grad():
            heatmaps = self.model(tensor)           # (1, 29, 512, 512)

        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()  # (29, 512, 512)

        # 3. Extract peaks (512-space)
        peaks_512 = self._extract_peaks(heatmaps_np)

        # 4. Scale to original resolution
        peaks_orig = self._scale_to_original(peaks_512, orig_size)

        # 5. Build landmarks dict
        landmarks = {
            name: [round(x, 1), round(y, 1)]
            for name, (x, y) in zip(LANDMARK_NAMES, peaks_orig)
        }

        # 6. Determine pixel size — calibrated override takes priority
        if override_pixel_size and override_pixel_size > 0:
            pixel_size = override_pixel_size
            logger.info(f"Using calibrated pixel_size: {pixel_size:.5f} mm/px")
        else:
            pixel_size = lookup_pixel_size(filename)

        # 7. Clinical analysis (angles + linear measurements)
        analysis = self._compute_analysis(peaks_orig, pixel_size)

        # 8. Full diagnostics table (Steiner + Soft Tissue norms)
        diagnostics_table = DiagnosticsEngine(peaks_orig, pixel_size).run()

        return {
            "landmarks": landmarks,
            "analysis": analysis,
            "diagnostics_table": diagnostics_table,
            "pixel_size": pixel_size,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smooth spline drawing helper  (module-level, called from CephPredictor.visualize)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_smooth_curve(
    img: np.ndarray,
    points: List[Optional[Tuple[int, int]]],
    color: Tuple[int, int, int],
    thickness: int = 2,
    n_interp: int = 200,
) -> None:
    """
    Draw a smooth PCHIP curve through *points* onto *img*.

    Uses PchipInterpolator (Piecewise Cubic Hermite Interpolating Polynomial)
    with cumulative chord-length parameterization.  PCHIP mathematically
    guarantees NO overshoot between data points, unlike standard cubic splines.

    Falls back to cv2.line segments if scipy is unavailable or < 3 valid points.
    """
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return

    if len(valid) < 3 or not _SCIPY_OK:
        for i in range(len(valid) - 1):
            cv2.line(img, valid[i], valid[i + 1], color, thickness, cv2.LINE_AA)
        return

    xs = np.array([p[0] for p in valid], dtype=np.float64)
    ys = np.array([p[1] for p in valid], dtype=np.float64)

    try:
        # Cumulative chord-length parameterization
        diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        t = np.zeros(len(valid))
        t[1:] = np.cumsum(diffs)
        if t[-1] == 0:
            return

        # PCHIP — monotone piecewise cubic, zero overshoot
        interp_x = PchipInterpolator(t, xs)
        interp_y = PchipInterpolator(t, ys)

        t_fine = np.linspace(0.0, t[-1], n_interp)
        x_fine = interp_x(t_fine)
        y_fine = interp_y(t_fine)

        curve_pts = np.column_stack(
            [np.round(x_fine).astype(np.int32),
             np.round(y_fine).astype(np.int32)]
        ).reshape((-1, 1, 2))
        cv2.polylines(img, [curve_pts], isClosed=False,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)
    except Exception as exc:
        logger.debug(f"PCHIP failed ({exc}) — using line fallback")
        for i in range(len(valid) - 1):
            cv2.line(img, valid[i], valid[i + 1], color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# visualize() — upgraded rendering layer (attached to CephPredictor below)
# ─────────────────────────────────────────────────────────────────────────────

# ── Landmark abbreviations ────────────────────────────────────────────────────
_LM_ABBREV: Dict[str, str] = {
    "A-point": "A",        "ANS": "ANS",          "B-point": "B",
    "Menton": "Me",        "Nasion": "N",          "Orbitale": "Or",
    "Pogonion": "Pog",     "PNS": "PNS",           "Pronasale": "Pn",
    "Ramus": "Ra",         "Sella": "S",            "Articulare": "Ar",
    "Condylion": "Co",     "Gnathion": "Gn",        "Gonion": "Go",
    "Porion": "Po",        "Lower 2nd PM Cusp": "L2", "Lower Incisor Tip": "LI",
    "Lower Molar Cusp": "LM", "Upper 2nd PM Cusp": "U2", "Upper Incisor Apex": "UIA",
    "Upper Incisor Tip": "UI", "Upper Molar Cusp": "UM", "Lower Incisor Apex": "LIA",
    "Labrale Inferius": "Li",  "Labrale Superius": "Ls",
    "Soft Tissue Nasion": "N'",  "Soft Tissue Pog": "Pg'",  "Subnasale": "Sn",
}

# ── Landmark type classification ──────────────────────────────────────────────
_SKELETAL_LM = {
    "A-point", "ANS", "B-point", "Menton", "Nasion", "Orbitale", "Pogonion",
    "PNS", "Ramus", "Sella", "Articulare", "Condylion", "Gnathion", "Gonion", "Porion",
}
_SOFT_TISSUE_LM = {
    "Pronasale", "Subnasale", "Labrale Superius", "Labrale Inferius",
    "Soft Tissue Nasion", "Soft Tissue Pog",
}
# Everything else → dental


def _ceph_visualize(self, image_bytes: bytes, filename: Optional[str] = None) -> bytes:
    """
    Run inference and return the original image annotated with the upgraded
    clinical rendering: color-coded dots, semi-transparent grouped construction
    lines, PCHIP anatomical splines, scale bar, and measurement readout.
    """
    # ── Run inference ────────────────────────────────────────────────────────
    result    = self.predict(image_bytes, filename=filename)
    landmarks: Dict = result["landmarks"]
    analysis:  Dict = result["analysis"]
    pixel_size: float = result.get("pixel_size", DEFAULT_PIXEL_SIZE)

    # ── Decode original image ─────────────────────────────────────────────
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image for visualisation.")

    h, w = img_bgr.shape[:2]

    # ── Scale-responsive sizes ────────────────────────────────────────────
    scale    = w / 1976.0          # normalise against a 1976-px reference
    dot_r    = max(10, int(11 * scale))  # 10-12px radius
    stroke_w = max(2, int(2.5 * scale))  # 2px stroke around dot
    lbl_scl  = max(0.65, 0.90 * scale)  # large enough to read at native res
    lbl_thk  = max(2, int(2.2 * scale))
    ln_t     = max(1, int(1.8 * scale))  # construction line thickness
    FONT     = cv2.FONT_HERSHEY_SIMPLEX

    # ── Helper: name → pixel tuple ────────────────────────────────────────
    def pt(name: str):
        c = landmarks.get(name)
        return (int(round(c[0])), int(round(c[1]))) if c else None

    # ─────────────────────────────────────────────────────────────────────
    # BGR color palette
    # ─────────────────────────────────────────────────────────────────────
    # Construction lines (semi-transparent, drawn on overlay)
    C_SAGITTAL = (255, 166,  77)   # rgba(77,166,255, 0.60) → blue   web
    C_VERTICAL = (114,  92, 255)   # rgba(255,92,114, 0.60) → red-pink
    C_DENTAL   = ( 77, 228, 255)   # rgba(255,228,77, 0.50) → yellow
    # Landmark dots
    D_SKEL_FILL  = (255, 166,  77)   # #4da6ff → blue  fill
    D_SKEL_STRK  = (255, 196, 135)   # #87c4ff → light-blue stroke
    D_SOFT_FILL  = (114,  92, 255)   # #ff5c72 → red-pink fill
    D_SOFT_STRK  = (163, 143, 255)   # #ff8fa3 → light-pink stroke
    D_DENT_FILL  = ( 77, 228, 255)   # #ffe44d → yellow fill
    D_DENT_STRK  = (153, 240, 255)   # #fff099 → pale-yellow stroke
    # Splines
    MINT   = (170, 255,   0)   # #00ffaa → soft tissue profile
    CYAN   = ( 80, 220, 100)   # teal    → mandibular outline
    # Labels & UI
    YELLOW = ( 77, 228, 255)   # #ffe44d
    WHITE  = (255, 255, 255)

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 0 — Semi-transparent construction lines
    # ─────────────────────────────────────────────────────────────────────
    overlay = img_bgr.copy()

    # Sagittal / horizontal reference planes (blue, α=0.60)
    sagittal_planes = [
        ("Porion",   "Orbitale"),        # Frankfort Horizontal
        ("Sella",    "Nasion"),           # SN Plane
        ("Gonion",   "Menton"),           # Mandibular Plane
        ("Nasion",   "A-point"),          # N-A arm
        ("Nasion",   "B-point"),          # N-B arm
    ]
    for a, b in sagittal_planes:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(overlay, pa, pb, C_SAGITTAL, ln_t, cv2.LINE_AA)

    # Vertical / measurement lines (red-pink, α=0.60)
    vertical_planes = [
        ("Nasion",    "Pogonion"),        # Facial Plane
        ("Sella",     "Gnathion"),        # Y-Axis
        ("A-point",   "Pogonion"),        # A-Po Line
        ("Condylion", "Gnathion"),        # Mandibular length Co-Gn
        ("Condylion", "A-point"),         # Maxillary length Co-A
    ]
    for a, b in vertical_planes:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(overlay, pa, pb, C_VERTICAL, ln_t, cv2.LINE_AA)

    # Dental / esthetic lines (yellow, α=0.50)
    dental_planes = [
        ("Pronasale",  "Soft Tissue Pog"),   # E-Line
    ]
    for a, b in dental_planes:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(overlay, pa, pb, C_DENTAL, ln_t, cv2.LINE_AA)

    # Blend the overlay once — increased opacity (was 0.62)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 0.5 — Anatomical bone contours (landmark-based PCHIP splines)
    # ─────────────────────────────────────────────────────────────────────
    # Instead of unreliable Canny edge detection, draw clean smooth curves
    # through the AI-predicted landmark positions — exactly how professional
    # cephalometric software renders bone tracings.
    BONE_CYAN = (255, 220, 10)   # BGR ≡ RGB(10, 220, 255) — bright cyan
    bone_t    = max(3, int(3.5 * scale))

    # 1. Mandibular outline:  Co → Ar → Ra → Go → Me → Gn → Pog → B
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Condylion"),
            pt("Articulare"),
            pt("Ramus"),
            pt("Gonion"),
            pt("Menton"),
            pt("Gnathion"),
            pt("Pogonion"),
            pt("B-point"),
        ],
        color=BONE_CYAN, thickness=bone_t,
    )

    # 2. Cranial base / upper skull:  S → N → Or → (Porion)
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Sella"),
            pt("Nasion"),
            pt("Orbitale"),
            pt("Porion"),
        ],
        color=BONE_CYAN, thickness=bone_t,
    )

    # 3. Maxillary arch:  S → PNS → ANS → A
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Sella"),
            pt("PNS"),
            pt("ANS"),
            pt("A-point"),
        ],
        color=BONE_CYAN, thickness=bone_t,
    )

    # 4. Upper dental arch:  Upper Molar → Upper 2nd PM → Upper Incisor Tip
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Upper Molar Cusp"),
            pt("Upper 2nd PM Cusp"),
            pt("Upper Incisor Apex"),
            pt("Upper Incisor Tip"),
        ],
        color=BONE_CYAN, thickness=max(2, bone_t - 1),
    )

    # 5. Lower dental arch:  Lower Molar → Lower PM → Lower Incisor Tip
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Lower Molar Cusp"),
            pt("Lower 2nd PM Cusp"),
            pt("Lower Incisor Apex"),
            pt("Lower Incisor Tip"),
        ],
        color=BONE_CYAN, thickness=max(2, bone_t - 1),
    )


    # ─────────────────────────────────────────────────────────────────────
    # LAYER 1 — PCHIP anatomical splines (opaque, drawn after blend)
    # ─────────────────────────────────────────────────────────────────────
    spline_t = max(3, int(3.0 * scale))  # 3px line width

    # Soft-tissue facial profile → mint #00ffaa
    # Order: N' → Pn → Sn → Ls (UL) → Li (LL) → Pg' → Gn → Me
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Soft Tissue Nasion"),
            pt("Pronasale"),
            pt("Subnasale"),
            pt("Labrale Superius"),
            pt("Labrale Inferius"),
            pt("Soft Tissue Pog"),
            pt("Gnathion"),
            pt("Menton"),
        ],
        color=MINT, thickness=spline_t,
    )

    # Mandibular bony outline → cyan/teal
    _draw_smooth_curve(
        img_bgr,
        [
            pt("Articulare"),
            pt("Gonion"),
            pt("Menton"),
            pt("Gnathion"),
            pt("Pogonion"),
        ],
        color=CYAN, thickness=spline_t,
    )

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 2+3 — Labels then Dots  (dots drawn LAST so always topmost)
    # Order: A compute positions → B pill backgrounds → D text/border → C dots
    # ─────────────────────────────────────────────────────────────────────
    pad_x = max(4, int(4 * scale))
    pad_y = max(3, int(3 * scale))
    edge_margin = int(0.12 * w)
    top_margin  = int(0.08 * h)
    BORDER_COL  = (77, 228, 255)   # subtle yellow border
    stroke_r    = dot_r + stroke_w

    # ── PASS A: compute all label positions ──────────────────────────────
    lbl_positions = {}
    for name, (x, y) in landmarks.items():
        cx, cy = int(round(x)), int(round(y))
        abbrev = _LM_ABBREV.get(name, name[:3])
        (tw, th), baseline = cv2.getTextSize(abbrev, FONT, lbl_scl, lbl_thk)

        # Offset: dx=+20px right, dy=-15px up from dot centre
        off_x = max(20, int(20 * scale))
        off_y = max(15, int(15 * scale))
        # Flip to left if too close to right edge
        if cx + off_x + tw > w - edge_margin:
            off_x = -(tw + off_x)
        # Flip below dot if too close to top edge
        if cy - off_y - th < top_margin:
            off_y = -(off_y + th + baseline)

        lbl_positions[name] = (cx + off_x, cy - off_y, tw, th, baseline)

    # ── PASS B: (Removed pill backgrounds to reduce clutter) ─────────────

    # ── PASS D: draw clean text labels ───────────────────────────
    for name, (tx, ty, tw, th, baseline) in lbl_positions.items():
        abbrev = _LM_ABBREV.get(name, name[:3])
        
        # Pure black text for maximum contrast against grey X-rays (BGR: 0, 0, 0)
        cv2.putText(img_bgr, abbrev, (tx, ty),
                    FONT, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # ── PASS C: draw landmark dots — always the topmost element ──────────
    for name, (x, y) in landmarks.items():
        cx, cy = int(round(x)), int(round(y))
        
        # Solid red pinpoint dots (BGR: 0, 0, 255)
        cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 4 — Scale bar (top-right corner)
    # ─────────────────────────────────────────────────────────────────────
    #   pixel_size is mm/px, so 10 mm = 10 / pixel_size pixels
    try:
        bar_mm  = 10.0
        bar_px  = int(round(bar_mm / pixel_size))
        bar_px  = max(20, min(bar_px, w // 3))   # clamp to sane range

        margin  = max(12, int(16 * scale))
        bar_y   = margin + int(8 * scale)
        bar_x1  = w - margin - bar_px
        bar_x2  = w - margin
        tick_h  = max(4, int(6 * scale))

        # Background pill for contrast
        pad = max(4, int(5 * scale))
        pill_tl = (bar_x1 - pad, bar_y - tick_h - pad)
        pill_br = (bar_x2 + pad, bar_y + pad + int(lbl_scl * 18))
        pill_ol = img_bgr.copy()
        cv2.rectangle(pill_ol, pill_tl, pill_br, (30, 30, 30), -1)
        cv2.addWeighted(pill_ol, 0.55, img_bgr, 0.45, 0, img_bgr)

        # Bar line + ticks
        bar_thk = max(1, int(1.5 * scale))
        cv2.line(img_bgr, (bar_x1, bar_y), (bar_x2, bar_y), YELLOW, bar_thk, cv2.LINE_AA)
        cv2.line(img_bgr, (bar_x1, bar_y - tick_h), (bar_x1, bar_y), YELLOW, bar_thk, cv2.LINE_AA)
        cv2.line(img_bgr, (bar_x2, bar_y - tick_h), (bar_x2, bar_y), YELLOW, bar_thk, cv2.LINE_AA)

        # Label centred above bar
        lbl_bar = f"{int(bar_mm)} mm"
        (tw, _th), _ = cv2.getTextSize(lbl_bar, FONT, lbl_scl, lbl_thk)
        lbl_bx = bar_x1 + (bar_px - tw) // 2
        lbl_by = bar_y - tick_h - max(2, int(3 * scale))
        cv2.putText(img_bgr, lbl_bar, (lbl_bx, lbl_by),
                    FONT, lbl_scl, (30, 30, 30), lbl_thk + 1, cv2.LINE_AA)
        cv2.putText(img_bgr, lbl_bar, (lbl_bx, lbl_by),
                    FONT, lbl_scl, YELLOW, lbl_thk, cv2.LINE_AA)
    except Exception:
        pass   # scale bar is cosmetic — never crash for it

    # ─────────────────────────────────────────────────────────────────────
    # LAYER 5 — Measurement readout (bottom-left, semi-transparent panel)
    # ─────────────────────────────────────────────────────────────────────
    meas_lines = [
        f"SNA: {analysis['SNA']:.1f} deg",
        f"SNB: {analysis['SNB']:.1f} deg",
        f"ANB: {analysis['ANB']:.1f} deg",
        f"MdL: {analysis.get('Mandibular Length (mm)', 0):.1f} mm",
        f"MxL: {analysis.get('Maxillary Length (mm)', 0):.1f} mm",
    ]
    meas_scl  = max(0.70, 0.95 * scale)
    meas_thk  = max(2, int(2.2 * scale))
    line_gap  = max(32, int(38 * scale))
    mx_margin = max(16, int(20 * scale))
    panel_h   = len(meas_lines) * line_gap + mx_margin * 2
    panel_w   = max(240, int(295 * scale))
    panel_y   = h - panel_h - mx_margin

    # Semi-transparent dark panel
    meas_bg = img_bgr.copy()
    cv2.rectangle(meas_bg, (mx_margin, panel_y),
                  (mx_margin + panel_w, h - mx_margin), (25, 25, 25), -1)
    cv2.addWeighted(meas_bg, 0.65, img_bgr, 0.35, 0, img_bgr)

    for i, txt in enumerate(meas_lines):
        ty = panel_y + mx_margin + (i + 1) * line_gap
        cv2.putText(img_bgr, txt, (mx_margin + 8, ty),
                    FONT, meas_scl, (30, 30, 30), meas_thk + 1, cv2.LINE_AA)
        cv2.putText(img_bgr, txt, (mx_margin + 8, ty),
                    FONT, meas_scl, YELLOW, meas_thk, cv2.LINE_AA)

    # ── Encode ────────────────────────────────────────────────────────────
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise RuntimeError("Failed to encode annotated image as JPEG.")
    return buf.tobytes()


# Attach visualize to CephPredictor as a proper bound method
CephPredictor.visualize = _ceph_visualize


# -----------------------------------------------------------------------------
# visualize_from_landmarks — bypass PyTorch, render from provided coords
# -----------------------------------------------------------------------------

def _visualize_from_landmarks(
    self,
    image_bytes: bytes,
    landmarks: Dict,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
) -> tuple:
    """
    Re-render cephalometric image from caller-supplied landmark coordinates.
    Skips PyTorch inference entirely.

    landmarks : dict  {name: [x, y]}  OR  {name: {"x": ..., "y": ...}}
    Returns   : (jpeg_bytes: bytes, diagnostics_table: list[dict])
    """
    # Normalise to {name: [x, y]}
    lm_norm: Dict = {}
    for name, val in landmarks.items():
        if isinstance(val, dict):
            lm_norm[name] = [float(val["x"]), float(val["y"])]
        else:
            lm_norm[name] = [float(val[0]), float(val[1])]

    # Reconstruct the ordered peaks_orig list expected by the ML engine
    peaks_orig = []
    from core.ml.inference import LANDMARK_NAMES
    for name in LANDMARK_NAMES:
        if name in lm_norm:
            peaks_orig.append(tuple(lm_norm[name]))
        else:
            peaks_orig.append((0.0, 0.0))  # fallback if missing

    # Compute analysis (angles/lengths)
    analysis = self._compute_analysis(peaks_orig, pixel_size)
    
    # Run diagnostics (pure geometry — no model)
    from core.ml.diagnostics import DiagnosticsEngine
    diagnostics_table = DiagnosticsEngine(peaks_orig, pixel_size).run()

    result_stub = {
        "landmarks":         lm_norm,
        "analysis":          analysis,
        "pixel_size":        pixel_size,
        "diagnostics_table": diagnostics_table,
    }

    # Temporarily patch self.predict so _ceph_visualize sees the stub
    _orig_predict = self.predict

    def _stub(img_b, **kw):
        return result_stub

    self.predict = _stub
    try:
        jpeg_bytes = _ceph_visualize(self, image_bytes)
    finally:
        self.predict = _orig_predict

    return jpeg_bytes, diagnostics_table


CephPredictor.visualize_from_landmarks = _visualize_from_landmarks
