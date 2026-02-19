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

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.ml.model import UNet

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

    def predict(self, image_bytes: bytes, filename: Optional[str] = None) -> Dict:
        """
        Full inference pipeline.

        Args:
            image_bytes: Raw bytes of a cephalogram image (JPG / PNG / BMP).
            filename: Original upload filename (used to look up pixel_size).

        Returns:
            {
                "landmarks": { "Sella": [x, y], ... },   # original-space coords
                "analysis":  { "SNA": 82.3, "SNB": 79.1, "ANB": 3.2 }
            }
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

        # 6. Look up pixel size for this image
        pixel_size = lookup_pixel_size(filename)

        # 7. Clinical analysis (angles + linear measurements)
        analysis = self._compute_analysis(peaks_orig, pixel_size)

        return {
            "landmarks": landmarks,
            "analysis": analysis,
            "pixel_size": pixel_size,
        }

    def visualize(self, image_bytes: bytes, filename: Optional[str] = None) -> bytes:
        """
        Run inference and return the original image annotated with
        landmark dots and angle measurements as JPEG bytes.

        Draws:
          • Red filled circle + white label for each of the 29 landmarks
          • Semi-transparent black banner in top-left with SNA / SNB / ANB
        """
        # ── Run inference ────────────────────────────────────────────────────
        result = self.predict(image_bytes, filename=filename)
        landmarks: Dict = result["landmarks"]
        analysis: Dict  = result["analysis"]

        # ── Decode original image (keep colour) ──────────────────────────────
        nparr   = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image for visualisation.")

        h, w = img_bgr.shape[:2]

        # Scale sizes relative to image resolution
        # Clinical images are typically 1800-2400 px wide
        dot_r    = max(4, w // 400)        # slightly bigger dot (4px)
        line_t   = max(1, w // 900)        # thin clinical line
        font_scl = max(0.35, w / 4000)
        thick    = max(1, w // 900)
        FONT     = cv2.FONT_HERSHEY_SIMPLEX

        # ── Helper: named landmark → integer (x, y) tuple ────────────────────
        def pt(name: str):
            coords = landmarks.get(name)
            if coords:
                return (int(round(coords[0])), int(round(coords[1])))
            return None

        # ── Color palette (BGR) ───────────────────────────────────────────────
        DARK_BLUE = (120,  60,   0)   # dark navy for angle planes
        RED_LINE  = (0,   30, 200)    # deep red for length lines
        RED_DOT   = (0,    0, 220)    # landmark dot color
        WHITE     = (255, 255, 255)

        # ── 1. Angle / plane lines (dark blue, thin, STRICT POINT TO POINT) ──
        angle_segs = [
            ("Sella",  "Nasion"),    # S-N cranial base plane
            ("Nasion", "A-point"),   # N-A  (SNA angle arm)
            ("Nasion", "B-point"),   # N-B  (SNB angle arm)
        ]
        for a, b in angle_segs:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                cv2.line(img_bgr, pa, pb, DARK_BLUE, line_t, cv2.LINE_AA)

        # ── 2. Frankfort Horizontal: Porion → Orbitale (STRICT POINT TO POINT)
        p_por, p_orb = pt("Porion"), pt("Orbitale")
        if p_por and p_orb:
            cv2.line(img_bgr, p_por, p_orb, DARK_BLUE, line_t, cv2.LINE_AA)

        # ── 3. Mandibular Plane: Gonion → Menton (STRICT POINT TO POINT) ─────
        p_go, p_me = pt("Gonion"), pt("Menton")
        if p_go and p_me:
            cv2.line(img_bgr, p_go, p_me, DARK_BLUE, line_t, cv2.LINE_AA)

        # ── 4. Length measurement lines (deep red, slightly thicker) ──────────
        length_segs = [
            ("Condylion", "Gnathion"),   # mandibular length Co-Gn
            ("Condylion", "A-point"),    # maxillary length Co-A
        ]
        for a, b in length_segs:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                cv2.line(img_bgr, pa, pb, RED_LINE, line_t + 1, cv2.LINE_AA)

        # ── 5. Landmark dots & labels (drawn last so they sit on top of lines)
        for name, (x, y) in landmarks.items():
            cx, cy = int(round(x)), int(round(y))
            # Draw the solid red dot
            cv2.circle(img_bgr, (cx, cy), dot_r, RED_DOT, -1, cv2.LINE_AA)
            # Draw the label (e.g., "S", "N", "A-p") offset by +6px
            lbl = name[:3]  # Keep it short
            cv2.putText(
                img_bgr, lbl,
                (cx + 6, cy + 6),
                FONT, font_scl * 0.9, WHITE, thick, cv2.LINE_AA
            )

        # ── 6. Stats banner (semi-transparent dark-grey, ASCII text) ──────────
        lines_txt = [
            f"SNA: {analysis['SNA']:.1f} deg",
            f"SNB: {analysis['SNB']:.1f} deg",
            f"ANB: {analysis['ANB']:.1f} deg",
            f"Mand: {analysis.get('Mandibular Length (mm)', 0):.1f} mm",
            f"Max:  {analysis.get('Maxillary Length (mm)', 0):.1f} mm",
        ]
        line_h     = int(28 * font_scl * 2.8)
        banner_h   = line_h * len(lines_txt) + 12
        banner_w   = int(260 * font_scl * 2.8)
        overlay    = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.60, img_bgr, 0.40, 0, img_bgr)

        for i, txt in enumerate(lines_txt):
            y_pos = 10 + int((i + 1) * line_h)
            cv2.putText(img_bgr, txt, (8, y_pos),
                        FONT, font_scl * 1.3, WHITE, thick, cv2.LINE_AA)

        # ── Encode to JPEG bytes ─────────────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Failed to encode annotated image as JPEG.")

        return buf.tobytes()
