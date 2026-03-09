"""
diagnostics.py — Cephalometric Clinical Math Engine

Calculates standard orthodontic metrics (Steiner, McNamara, Soft Tissue)
from predicted landmark coordinates and compares them to published norms.

Usage:
    from core.ml.diagnostics import DiagnosticsEngine
    table = DiagnosticsEngine(peaks_orig, pixel_size).run()
"""

import math
from typing import Any, Dict, List, Tuple

Point = Tuple[float, float]

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vec(origin: Point, tip: Point) -> Tuple[float, float]:
    return tip[0] - origin[0], tip[1] - origin[1]


def _angle_at_vertex(p1: Point, vertex: Point, p2: Point) -> float:
    """Angle (degrees) at `vertex` in the triangle p1-vertex-p2."""
    v1 = _vec(vertex, p1)
    v2 = _vec(vertex, p2)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _angle_between_lines(a1: Point, a2: Point, b1: Point, b2: Point) -> float:
    """Acute angle (degrees) between line a1→a2 and line b1→b2."""
    va = _vec(a1, a2)
    vb = _vec(b1, b2)
    dot = va[0]*vb[0] + va[1]*vb[1]
    ma = math.sqrt(va[0]**2 + va[1]**2)
    mb = math.sqrt(vb[0]**2 + vb[1]**2)
    if ma == 0 or mb == 0:
        return 0.0
    cos_theta = abs(dot) / (ma * mb)      # abs → acute angle
    return math.degrees(math.acos(min(1.0, cos_theta)))


def _signed_dist_point_to_line_mm(
    point: Point, line_p1: Point, line_p2: Point, pixel_size: float
) -> float:
    """
    Signed perpendicular distance from `point` to the infinite line through
    line_p1 and line_p2, converted to mm.

    Sign convention:
      Positive  → point is to the LEFT of the direction line_p1→line_p2
      Negative  → point is to the RIGHT
    For the E-line (Pronasale→Soft-Tissue-Pog pointing downwards on a lateral
    ceph), negative values mean the lip is POSTERIOR to the E-line (retrusive),
    which matches the standard clinical convention.
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return 0.0
    # Cross-product / length = signed distance in pixels
    signed_px = ((point[0] - line_p1[0]) * dy - (point[1] - line_p1[1]) * dx) / length
    return signed_px * pixel_size


# ─────────────────────────────────────────────────────────────────────────────
# Landmark indices (mirror LANDMARK_NAMES in inference.py)
# ─────────────────────────────────────────────────────────────────────────────
_A           = 0
_ANS         = 1
_B           = 2
_MENTON      = 3
_NASION      = 4
_ORBITALE    = 5
_POG         = 6
_SELLA       = 10
_CONDYLION   = 12
_GNATHION    = 13
_GONION      = 14
_PORION      = 15
_LAB_INF     = 24   # Labrale Inferius (lower lip)
_LAB_SUP     = 25   # Labrale Superius (upper lip)
_PRONASALE   = 8    # Tip of nose
_ST_POG      = 27   # Soft-Tissue Pogonion (E-line endpoint)
_UI_TIP      = 21   # Upper Incisor Tip
_UI_APEX     = 20   # Upper Incisor Apex
_LI_TIP      = 17   # Lower Incisor Tip
_LI_APEX     = 23   # Lower Incisor Apex


# ─────────────────────────────────────────────────────────────────────────────
# Norm definition helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_row(
    parameter: str,
    value: float,
    norm_mean: float,
    norm_range: float,      # ± half-width
    unit: str,
    comment_high: str,
    comment_low: str,
    comment_normal: str = "Within normal range",
    fmt: str = ".1f",
) -> Dict[str, Any]:
    diff = value - norm_mean
    is_abnormal = abs(diff) > norm_range
    if is_abnormal:
        comment = comment_high if diff > 0 else comment_low
    else:
        comment = comment_normal

    ref_str  = f"{norm_mean:.0f}±{norm_range:.0f}{unit}"
    val_str  = f"{value:{fmt}}{unit}"
    diff_str = f"{'+' if diff >= 0 else ''}{diff:{fmt}}{unit}"

    return {
        "parameter":   parameter,
        "value":       val_str,
        "reference":   ref_str,
        "diff":        diff_str,
        "comment":     comment,
        "is_abnormal": is_abnormal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic engine
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsEngine:
    """
    Accepts the 29 predicted landmark coordinates (original-image pixel space)
    and the pixel_size (mm/px) and computes a structured diagnostics table.
    """

    def __init__(
        self,
        peaks: List[Tuple[float, float]],
        pixel_size: float,
    ):
        self.lm = peaks
        self.ps = pixel_size

    def _pt(self, idx: int) -> Point:
        return self.lm[idx]

    def run(self) -> List[Dict[str, Any]]:
        lm = self.lm
        ps = self.ps
        rows: List[Dict[str, Any]] = []

        # ── 1. SNA Angle ──────────────────────────────────────────────────────
        sna = _angle_at_vertex(lm[_SELLA], lm[_NASION], lm[_A])
        rows.append(_build_row(
            "SNA", sna, norm_mean=82, norm_range=2, unit="°",
            comment_high="Maxillary protrusion",
            comment_low="Maxillary retrusion",
        ))

        # ── 2. SNB Angle ──────────────────────────────────────────────────────
        snb = _angle_at_vertex(lm[_SELLA], lm[_NASION], lm[_B])
        rows.append(_build_row(
            "SNB", snb, norm_mean=80, norm_range=2, unit="°",
            comment_high="Mandibular protrusion",
            comment_low="Mandibular retrusion",
        ))

        # ── 3. ANB Angle ──────────────────────────────────────────────────────
        anb = sna - snb
        rows.append(_build_row(
            "ANB", anb, norm_mean=2, norm_range=2, unit="°",
            comment_high="Class II skeletal pattern",
            comment_low="Class III skeletal pattern",
        ))

        # ── 4. SN-GoGn (Mandibular Plane Angle) ──────────────────────────────
        sn_gogn = _angle_between_lines(
            lm[_SELLA], lm[_NASION],
            lm[_GONION], lm[_MENTON],
        )
        rows.append(_build_row(
            "SN-GoGn", sn_gogn, norm_mean=32, norm_range=4, unit="°",
            comment_high="Vertical growth / Open bite tendency",
            comment_low="Horizontal growth / Deep bite tendency",
        ))

        # ── 5. Upper Lip to E-Line ────────────────────────────────────────────
        # E-line: Pronasale → Soft Tissue Pogonion (tip of nose → chin)
        ul_e = _signed_dist_point_to_line_mm(
            lm[_LAB_SUP], lm[_PRONASALE], lm[_ST_POG], ps
        )
        # Flip sign: on a lateral ceph, the face points right so "in front of
        # E-line" (protrusion) comes out negative from the cross-product.
        # Negate so positive = protrusive, negative = retrusive (clinical conv.)
        ul_e = -ul_e
        rows.append(_build_row(
            "Upper Lip – E-Line", ul_e,
            norm_mean=-4, norm_range=2, unit="mm",
            comment_high="Protrusive upper lip",
            comment_low="Retrusive upper lip",
        ))

        # ── 6. Lower Lip to E-Line ────────────────────────────────────────────
        ll_e = _signed_dist_point_to_line_mm(
            lm[_LAB_INF], lm[_PRONASALE], lm[_ST_POG], ps
        )
        ll_e = -ll_e
        rows.append(_build_row(
            "Lower Lip - E-Line", ll_e,
            norm_mean=-2, norm_range=2, unit="mm",
            comment_high="Protrusive lower lip",
            comment_low="Retrusive lower lip",
        ))

        # ── 7. Maxillary Length / McNamara (Co-A) ─────────────────────────────
        def _dist_mm(p1: Point, p2: Point) -> float:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            return math.sqrt(dx * dx + dy * dy) * ps

        try:
            max_len = _dist_mm(lm[_CONDYLION], lm[_A])
            rows.append(_build_row(
                "Maxillary Length (Co-A)", max_len,
                norm_mean=90, norm_range=5, unit="mm",
                comment_high="Maxillary hyperplasia",
                comment_low="Maxillary hypoplasia",
            ))
            print(f"[diagnostics] metric 7 OK: Maxillary Length = {max_len:.1f} mm")
        except Exception as exc:
            print(f"[diagnostics] metric 7 FAILED: {exc}")
            rows.append({"parameter": "Maxillary Length (Co-A)", "value": "Error",
                         "reference": "90±5mm", "diff": "N/A",
                         "comment": f"Calculation error: {exc}", "is_abnormal": True})

        # ── 8. Mandibular Length / McNamara (Co-Gn) ──────────────────────────
        try:
            mand_len = _dist_mm(lm[_CONDYLION], lm[_GNATHION])
            rows.append(_build_row(
                "Mandibular Length (Co-Gn)", mand_len,
                norm_mean=120, norm_range=5, unit="mm",
                comment_high="Mandibular hyperplasia",
                comment_low="Mandibular hypoplasia",
            ))
            print(f"[diagnostics] metric 8 OK: Mandibular Length = {mand_len:.1f} mm")
        except Exception as exc:
            print(f"[diagnostics] metric 8 FAILED: {exc}")
            rows.append({"parameter": "Mandibular Length (Co-Gn)", "value": "Error",
                         "reference": "120±5mm", "diff": "N/A",
                         "comment": f"Calculation error: {exc}", "is_abnormal": True})

        # ── 9. Upper Incisor to NA (Dental Steiner) ───────────────────────────
        try:
            ui_na = _angle_between_lines(
                lm[_UI_TIP], lm[_UI_APEX],   # upper incisor long axis
                lm[_NASION], lm[_A],          # NA reference line
            )
            rows.append(_build_row(
                "UI to NA", ui_na,
                norm_mean=22, norm_range=2, unit="deg",
                comment_high="Proclined upper incisors",
                comment_low="Retroclined upper incisors",
            ))
            print(f"[diagnostics] metric 9 OK: UI to NA = {ui_na:.1f} deg")
        except Exception as exc:
            print(f"[diagnostics] metric 9 FAILED: {exc}")
            rows.append({"parameter": "UI to NA", "value": "Error",
                         "reference": "22±2deg", "diff": "N/A",
                         "comment": f"Calculation error: {exc}", "is_abnormal": True})

        # ── 10. Lower Incisor to NB (Dental Steiner) ─────────────────────────
        try:
            li_nb = _angle_between_lines(
                lm[_LI_TIP], lm[_LI_APEX],   # lower incisor long axis
                lm[_NASION], lm[_B],          # NB reference line
            )
            rows.append(_build_row(
                "LI to NB", li_nb,
                norm_mean=25, norm_range=2, unit="deg",
                comment_high="Proclined lower incisors",
                comment_low="Retroclined lower incisors",
            ))
            print(f"[diagnostics] metric 10 OK: LI to NB = {li_nb:.1f} deg")
        except Exception as exc:
            print(f"[diagnostics] metric 10 FAILED: {exc}")
            rows.append({"parameter": "LI to NB", "value": "Error",
                         "reference": "25±2deg", "diff": "N/A",
                         "comment": f"Calculation error: {exc}", "is_abnormal": True})

        print(f"[diagnostics] run() complete — returning {len(rows)} rows")
        return rows
