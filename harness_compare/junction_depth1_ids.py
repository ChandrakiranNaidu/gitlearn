#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 - Junction detection and depth-1 junction identities.

This file is also the shared vector-extraction engine that Script 2
(build_logical_segments.py) imports.

Pipeline
--------
    Script 1  junction_depth1_ids.py              (this file)
    Script 2  build_logical_segments.py           (imports this file)
    Script 3  compare_and_render_segments_v6.py   (stand-alone)

What this script does for ONE page of ONE PDF
---------------------------------------------
1.  Reads the original PDF vector drawings with PyMuPDF.
2.  Detects circular objects from their Bezier structure, size and aspect ratio,
    merges concentric duplicates into one physical symbol and keeps
    "contains a filled circle" plus the outer diameter.
3.  Extracts harness wires: stroked line vectors whose stroke width equals the
    harness wire width (fixed value or auto-detected).
4.  Extracts every text object with value, centre, bounding box, direction,
    orientation (horizontal / vertical / angled) and numeric flag.
5.  For every circle, finds the wire vectors touching or crossing it, computes
    the unique leave directions and classifies it as
        junction_dot (3+ directions), two_direction_dot, endpoint_dot,
        isolated_dot, hollow_symbol, large_hollow_element, large_solid_element.
6.  Builds a connectivity graph (merged wire endpoints = vertices,
    wire vectors = edges).
7.  For every true junction, traces each outgoing direction independently
    through ordinary bends, and stops at the first valid numeric wire label,
    a substantial circular element, another branch, a wire end or the trace
    limit.  Only labels that obey the strict placement convention are accepted:
        horizontal wire piece -> horizontal numeric text BELOW the wire
        vertical wire piece   -> vertical (rotated) numeric text RIGHT of the wire
        angled wire piece     -> never claims a label
8.  Builds a position-independent identity such as DOT1[125|208|430]
    (UNLABELLED for directions without a confident label).
9.  Writes a junction JSON file and diagnostic images.

Coordinates
-----------
All geometry is converted to the DISPLAYED page frame (page /Rotate applied,
origin top-left, y grows downwards).  A page rendered with
page.get_pixmap(matrix=Matrix(Z, Z)) maps a point (x, y) to pixel (x*Z, y*Z).

Example
-------
    python junction_depth1_ids.py --pdf old.pdf --out junctions_old --page 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # PyMuPDF >= 1.24 exposes "pymupdf"; older versions only "fitz".
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

ENGINE_NAME = "harness-vector-engine"
ENGINE_VERSION = "1.0"

Point = Tuple[float, float]

# ---------------------------------------------------------------------------
# Text classification helpers (shared with Script 2)
# ---------------------------------------------------------------------------

# A wire number is digits with an optional decimal component: 52, 52.0, 35.8
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Generic terminal size / route codes that must never be a terminal identity:
# 1M, 2M, 8M, 2M,8M, 2M/4M
GENERIC_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?M(?:\s*[,/]\s*\d+(?:\.\d+)?M)*$", re.IGNORECASE)

# Wire / part specification codes that are not terminal names:
# EN6049-007, NSA935805-06, ABS2418A05-5, D924-95612-051
SPEC_CODE_RES = (
    re.compile(r"^[A-Z]{2,5}\d{3,}[A-Z0-9]*(?:-[A-Z0-9]+)+$"),
    re.compile(r"^[A-Z]\d{3,}(?:-\d{2,})+$"),
)

# Text-derived annotation keywords.  Their changes are review items, never
# confirmed physical changes.
ANNOTATION_WORDS = (
    "CLAMP", "MARKER", "ROUTE", "LABEL", "BRAID", "SUPPORT", "SPLIT SUPPORT",
    "SLEEVE", "TIE", "TAPE", "GROMMET", "SHIELD",
)
ANNOTATION_INNER_WORDS = ("MARKER", "LABEL", "ROUTE")


def normalize_text(raw: str) -> str:
    """Collapse whitespace; numeric values such as '52.0' become '52'."""
    s = " ".join(str(raw).split())
    if NUMERIC_RE.match(s):
        if "." in s:
            whole, frac = s.split(".", 1)
            if set(frac) <= {"0"}:
                s = whole
    return s


def is_numeric_text(norm: str) -> bool:
    return bool(NUMERIC_RE.match(norm))


def is_generic_size(norm: str) -> bool:
    return bool(GENERIC_SIZE_RE.match(norm.replace(" ", "")))


def is_spec_code(norm: str) -> bool:
    s = norm.replace(" ", "").upper()
    return any(r.match(s) for r in SPEC_CODE_RES)


def annotation_keyword(norm: str) -> Optional[str]:
    """Return the annotation keyword contained in a text, if any."""
    up = norm.upper()
    if "SPLIT SUPPORT" in up:
        return "SUPPORT"
    for w in ANNOTATION_WORDS:
        if re.search(r"(?<![A-Z])" + re.escape(w) + r"(?![A-Z])", up):
            return "SUPPORT" if w == "SPLIT SUPPORT" else w
    return None


def is_meaningful_identifier(norm: str) -> bool:
    """True for terminal / element names such as 7554VC-A, 6711VN/B2, 14MP-A, B1."""
    s = norm.strip()
    if len(s) < 2 or is_numeric_text(s) or is_generic_size(s) or is_spec_code(s):
        return False
    if annotation_keyword(s):
        return False
    return bool(re.search(r"[A-Za-z]", s)) and bool(re.search(r"\d", s))


def numeric_sort_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Values measured on the reference drawing the rules were tuned on.  Every distance
# threshold below is stored for that drawing and rescaled per PDF by derive_config()
# from the units measured on the PDF itself (see calibrate_profile()).
REF_TEXT_UNIT = 5.39      # median font size of numeric labels (pt)
REF_WIRE_UNIT = 0.539     # harness wire stroke width (pt)
REF_DOT_UNIT = 2.16       # junction dot outer diameter (pt)
REF_LABEL_DISTANCE = 5.04 # median distance of a wire number's centre from its wire (pt)


@dataclass
class EngineConfig:
    # Calibration (None = measured per PDF)
    wire_width: Optional[float] = None       # explicit harness wire stroke width, else auto
    wire_widths: Optional[Tuple[float, ...]] = None
    label_max_side_override: Optional[float] = None
    side_h: Optional[int] = None             # +1 labels below horizontal wires, -1 above, 0 on the wire
    side_v: Optional[int] = None             # +1 labels right of vertical wires, -1 left, 0 on the wire
    deskew: bool = True
    text_unit: float = REF_TEXT_UNIT
    wire_unit: float = REF_WIRE_UNIT
    dot_unit: float = REF_DOT_UNIT
    layout_unit: float = REF_TEXT_UNIT
    # Wire extraction
    wire_width_tol: float = 0.03
    min_wire_length: float = 0.25
    significant_min_width: float = 0.22     # non-wire lines thinner AND shorter than this are texture
    significant_min_length: float = 3.0
    gap_bridge_max: float = 6.0
    # Graph construction
    endpoint_merge_tol: float = 0.75
    split_tol: float = 0.5
    # Circle detection / classification
    circle_roundness_tol: float = 0.12
    circle_merge_tol: float = 0.6
    dot_max_diameter: float = 3.0
    element_min_diameter: float = 4.0
    circle_touch_tol: float = 0.8
    direction_merge_deg: float = 12.0
    # Orientation
    axis_tol_deg: float = 8.0               # wire piece horizontal / vertical tolerance
    text_axis_tol_deg: float = 12.0         # text horizontal / vertical tolerance
    # Strict label placement
    label_min_side: float = 0.4             # text centre must be at least this far on the required side
    label_max_side: float = 10.5            # ... and at most this far
    label_overlap_tol: float = 0.6          # text box may overlap the wire by at most this
    projection_margin: float = 2.0          # text centre may project this far beyond a piece end
    inline_max_side: float = 3.0            # labels printed on the wire (side convention 0)
    centre_weight: float = 0.5
    # Depth-1 tracing
    trace_max_length: float = 2500.0
    trace_max_steps: int = 400
    ambiguity_score_gap: float = 0.08
    # Terminal detection
    terminal_symbol_tol: float = 4.0
    terminal_symbol_min_width: float = 0.3
    terminal_text_radius: float = 30.0
    element_text_radius: float = 22.0
    # Text splitting
    text_gap_factor: float = 0.8
    # Sheet border (drawing frame, zone markers, title-block tables)
    border_detection: bool = True
    border_min_fraction: float = 0.6       # frame rectangle covers at least this share of page width / height
    border_corner_tol: float = 2.5
    border_band_tol: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Distance from p to segment ab and the clamped parameter t in [0, 1]."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return dist(p, a), 0.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)), t


def unit(v: Sequence[float]) -> Tuple[float, float]:
    L = math.hypot(v[0], v[1])
    if L <= 1e-12:
        return (0.0, 0.0)
    return (v[0] / L, v[1] / L)


def angle_deg(v: Sequence[float]) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def piece_orientation(p0: Sequence[float], p1: Sequence[float], tol_deg: float) -> str:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) <= 1e-9:
        return "point"
    a = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    if a <= tol_deg or a >= 180.0 - tol_deg:
        return "horizontal"
    if abs(a - 90.0) <= tol_deg:
        return "vertical"
    return "angled"


def rect_distance(p: Sequence[float], r: Sequence[float]) -> float:
    """Distance from point p to rectangle r = (x0, y0, x1, y1); 0 inside."""
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def rnd(v: float, nd: int = 3) -> float:
    return float(round(float(v), nd))


def rpt(p: Sequence[float], nd: int = 3) -> List[float]:
    return [rnd(p[0], nd), rnd(p[1], nd)]


class GridIndex:
    """Tiny uniform-grid spatial index for bounding boxes."""

    def __init__(self, cell: float = 25.0):
        self.cell = float(cell)
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def _cells(self, x0: float, y0: float, x1: float, y1: float):
        c = self.cell
        for gx in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for gy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                yield (gx, gy)

    def insert(self, key: int, x0: float, y0: float, x1: float, y1: float) -> None:
        for cell in self._cells(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)):
            self.buckets[cell].append(key)

    def query(self, x0: float, y0: float, x1: float, y1: float) -> List[int]:
        seen = set()
        out = []
        for cell in self._cells(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)):
            for k in self.buckets.get(cell, ()):
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_pdf_page(path: str, page_no: int):
    """Open a PDF and return (doc, page).  The caller must close doc."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(path)
    if page_no < 0 or page_no >= doc.page_count:
        n = doc.page_count
        doc.close()
        raise ValueError(f"Page {page_no} out of range for {path} ({n} page(s), zero-based)")
    return doc, doc[page_no]


def source_pdf_info(path: str, page) -> dict:
    return {
        "path": os.path.abspath(path),
        "file_name": os.path.basename(path),
        "sha256": sha256_file(path),
        "page": int(page.number),
        "page_width": rnd(page.rect.width),
        "page_height": rnd(page.rect.height),
        "rotation": int(page.rotation),
    }


def page_matrix(page) -> "fitz.Matrix":
    """Unrotated page coordinates -> displayed page coordinates."""
    return page.rotation_matrix


def to_view(M, p) -> Tuple[float, float]:
    """Affine transform with plain floats (much faster than fitz.Point * Matrix)."""
    x, y = float(p[0]), float(p[1])
    return (M.a * x + M.c * y + M.e, M.b * x + M.d * y + M.f)


def vec_to_view(M, v: Sequence[float]) -> Tuple[float, float]:
    return (M.a * v[0] + M.c * v[1], M.b * v[0] + M.d * v[1])


def render_page_bgr(page, zoom: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _char_extent(bbox: "fitz.Rect", d: Sequence[float]) -> float:
    return abs(d[0]) * bbox.width + abs(d[1]) * bbox.height


def _split_span_chars(chars: List[dict], d: Sequence[float], size: float, gap_factor: float) -> List[List[dict]]:
    """Split a span into words wherever the gap between visible characters is large."""
    groups: List[List[dict]] = []
    cur: List[dict] = []
    prev = None
    for ch in chars:
        c = ch.get("c", "")
        if not c.strip():
            if cur:
                cur.append(ch)
            continue
        r = fitz.Rect(ch["bbox"])
        if prev is not None:
            pr = fitz.Rect(prev["bbox"])
            pc = ((pr.x0 + pr.x1) / 2, (pr.y0 + pr.y1) / 2)
            cc = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            along = (cc[0] - pc[0]) * d[0] + (cc[1] - pc[1]) * d[1]
            gap = along - (_char_extent(pr, d) + _char_extent(r, d)) / 2.0
            if gap > gap_factor * max(size, 1.0) or along < -0.5 * max(size, 1.0):
                groups.append(cur)
                cur = []
        cur.append(ch)
        prev = ch
    if cur:
        groups.append(cur)
    return groups


def extract_text_objects(page, M, cfg: EngineConfig, visible: Optional["fitz.Rect"] = None) -> List[dict]:
    """Extract text objects (word groups) in the displayed frame.

    Characters are read without clipping and a whole word is kept when any part of it
    is on the visible page, so a name cut by the crop box is not shortened ('174VN' for
    '6174VN'), while text of other sheets placed off the page is ignored.
    """
    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
    raw = page.get_text("rawdict", flags=flags, clip=fitz.INFINITE_RECT())
    visible = visible if visible is not None else page.rect
    out: List[dict] = []
    tol = math.sin(math.radians(cfg.text_axis_tol_deg))
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            ld = line.get("dir", (1.0, 0.0))
            vd = unit(vec_to_view(M, ld))
            for span in line.get("spans", []):
                if not (fitz.Rect(span["bbox"]) * M).intersects(visible):
                    continue                  # text of another sheet placed off the page
                chars = span.get("chars") or []
                size = float(span.get("size", 0.0))
                for group in _split_span_chars(chars, ld, size, cfg.text_gap_factor):
                    raw_text = "".join(c.get("c", "") for c in group)
                    norm = normalize_text(raw_text)
                    if not norm:
                        continue
                    vis = [c for c in group if c.get("c", "").strip()]
                    rect = fitz.Rect()
                    for c in vis:
                        rect |= fitz.Rect(c["bbox"]) * M
                    if rect.is_empty and vis:
                        rect = fitz.Rect(vis[0]["bbox"]) * M
                    if not rect.intersects(visible):
                        continue
                    cx, cy = (rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0
                    if abs(vd[1]) <= tol:
                        orientation = "horizontal"
                    elif abs(vd[0]) <= tol:
                        orientation = "vertical"
                    else:
                        orientation = "angled"
                    out.append({
                        "id": "",
                        "text": raw_text.strip(),
                        "norm": norm,
                        "center": [rnd(cx), rnd(cy)],
                        "bbox": [rnd(rect.x0), rnd(rect.y0), rnd(rect.x1), rnd(rect.y1)],
                        "dir": [rnd(vd[0], 4), rnd(vd[1], 4)],
                        "angle_deg": rnd(angle_deg(vd), 2),
                        "orientation": orientation,
                        "is_numeric": is_numeric_text(norm),
                        "font_size": rnd(size, 2),
                    })
    out.sort(key=lambda t: (t["center"][1], t["center"][0], t["norm"]))
    for i, t in enumerate(out):
        t["id"] = f"T{i:05d}"
    return out


# ---------------------------------------------------------------------------
# Vector extraction
# ---------------------------------------------------------------------------

def _is_white(color) -> bool:
    if color is None:
        return False
    try:
        return all(float(c) >= 0.97 for c in color)
    except TypeError:
        return False


def _is_dashed(dashes) -> bool:
    if not dashes:
        return False
    s = str(dashes).strip()
    if s in ("[] 0", "[]0", "[] 0.0", "()", "[]"):
        return False
    return bool(re.search(r"\[\s*[\d.]", s))


def _bezier_points(p0, p1, p2, p3, n: int = 8) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0]
        y = mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts


def _path_items_view(items, M):
    """Yield (kind, [points in view frame]) for every path item."""
    for it in items:
        kind = it[0]
        if kind == "l":
            yield "l", [to_view(M, it[1]), to_view(M, it[2])]
        elif kind == "c":
            yield "c", [to_view(M, it[1]), to_view(M, it[2]), to_view(M, it[3]), to_view(M, it[4])]
        elif kind == "re":
            r = fitz.Rect(it[1])
            yield "poly", [to_view(M, r.tl), to_view(M, r.tr), to_view(M, r.br), to_view(M, r.bl), to_view(M, r.tl)]
        elif kind == "qu":
            q = it[1]
            yield "poly", [to_view(M, q.ul), to_view(M, q.ur), to_view(M, q.lr), to_view(M, q.ll), to_view(M, q.ul)]


def _circle_from_items(view_items, cfg: EngineConfig) -> Optional[Tuple[Tuple[float, float], float]]:
    """Return (centre, diameter) when a path is a (near) perfect circle."""
    curves = [pts for k, pts in view_items if k == "c"]
    if len(curves) < 2:
        return None
    for k, pts in view_items:
        if k == "l" and dist(pts[0], pts[1]) > 0.05:
            return None
        if k == "poly":
            return None
    samples: List[Tuple[float, float]] = []
    for c in curves:
        samples.extend(_bezier_points(*c, n=6))
    if dist(curves[0][0], curves[-1][3]) > 0.15 * max(1e-6, max(dist(samples[0], s) for s in samples)):
        return None  # not closed
    xs = [p[0] for p in samples]
    ys = [p[1] for p in samples]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w <= 0.02 * cfg.text_unit or h <= 0.02 * cfg.text_unit:
        return None
    if abs(w / h - 1.0) > 0.2:
        return None
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    radii = [dist((cx, cy), p) for p in samples]
    mean_r = sum(radii) / len(radii)
    if mean_r <= 0.05 or (max(radii) - min(radii)) / mean_r > 2 * cfg.circle_roundness_tol:
        return None
    return (cx, cy), 2.0 * mean_r


def _view_segments(view_items) -> List[Tuple[Tuple[float, float], Tuple[float, float], int, str]]:
    segs = []
    for idx, (k, pts) in enumerate(view_items):
        if k == "l":
            segs.append((pts[0], pts[1], idx, "l"))
        elif k == "c":
            b = _bezier_points(*pts, n=4)
            for a, c in zip(b, b[1:]):
                segs.append((a, c, idx, "c"))
        elif k == "poly":
            for a, c in zip(pts, pts[1:]):
                segs.append((a, c, idx, "poly"))
    return segs


def clip_segment(a: Sequence[float], b: Sequence[float], r) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Liang-Barsky clip of segment ab to rectangle r; None when fully outside."""
    x0, y0 = a
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r.x0), (dx, r.x1 - x0), (-dy, y0 - r.y0), (dy, r.y1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return ((x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy))


# ---------------------------------------------------------------------------
# Sheet border: drawing frame, zone markers, title-block tables
# ---------------------------------------------------------------------------

def _merge_collinear(lines: List[Tuple[float, float, float]], ctol: float = 0.6,
                     gap: float = 2.0) -> List[Tuple[float, float, float]]:
    """(lo, hi, c) axis-aligned lines -> merged extents per coordinate c, longest first."""
    out: List[List[float]] = []
    for lo, hi, c in sorted(lines, key=lambda l: (l[2], l[0])):
        for m in out:
            if abs(m[2] - c) <= ctol and lo <= m[1] + gap and hi >= m[0] - gap:
                m[0], m[1] = min(m[0], lo), max(m[1], hi)
                break
        else:
            out.append([lo, hi, c])
    out.sort(key=lambda m: -(m[1] - m[0]))
    return [tuple(m) for m in out]


def detect_sheet_border(page, M, drawings: list, text_centres: List[Tuple[float, float]],
                        cfg: EngineConfig) -> dict:
    """Drawing frame(s): large axis-aligned rectangles framing the sheet, any stroke width.

    A frame may be one rectangle path or four separate lines and may run partly off the
    page.  It must cover >= border_min_fraction of the page in both directions, be at most
    about one page in size (multi-sheet PDFs place other sheets off the page), overlap at
    least half the page and enclose most of the page text.  Returns the frames (outermost
    first, displayed frame) and the innermost one, which bounds the drawing area.
    """
    W, H = page.rect.width, page.rect.height
    if not cfg.border_detection:
        return {"frames": [], "inner": None}
    U = cfg.layout_unit
    ax_tol = max(0.5 * U / REF_TEXT_UNIT, 0.05)
    rects, hs, vs = [], [], []
    for d in drawings:
        if "s" not in (d.get("type") or "") or _is_white(d.get("color")):
            continue
        for kind, pts in _path_items_view(d.get("items", []), M):
            if kind == "poly":
                segs = list(zip(pts, pts[1:]))
                if all(abs(a[0] - b[0]) <= ax_tol or abs(a[1] - b[1]) <= ax_tol for a, b in segs):
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    rects.append((min(xs), min(ys), max(xs), max(ys)))
            elif kind == "l":
                segs = [(pts[0], pts[1])]
            else:
                continue
            for a, b in segs:
                if abs(a[1] - b[1]) <= ax_tol and abs(a[0] - b[0]) >= 0.5 * W:
                    hs.append((min(a[0], b[0]), max(a[0], b[0]), (a[1] + b[1]) / 2))
                elif abs(a[0] - b[0]) <= ax_tol and abs(a[1] - b[1]) >= 0.5 * H:
                    vs.append((min(a[1], b[1]), max(a[1], b[1]), (a[0] + b[0]) / 2))
    hs, vs = _merge_collinear(hs, 0.12 * U, 0.4 * U)[:16], _merge_collinear(vs, 0.12 * U, 0.4 * U)[:16]
    t = cfg.border_corner_tol
    for top in hs:
        for bot in hs:
            if bot[2] - top[2] < cfg.border_min_fraction * H:
                continue
            for left in vs:
                for right in vs:
                    x0, x1, y0, y1 = left[2], right[2], top[2], bot[2]
                    if x1 - x0 < cfg.border_min_fraction * W:
                        continue
                    if all(l[0] <= a + t and l[1] >= b - t for l, a, b in
                           ((top, x0, x1), (bot, x0, x1), (left, y0, y1), (right, y0, y1))):
                        rects.append((x0, y0, x1, y1))
    frames: List[Tuple[float, float, float, float]] = []
    for r in rects:
        w, h = r[2] - r[0], r[3] - r[1]
        if w < cfg.border_min_fraction * W or h < cfg.border_min_fraction * H:
            continue
        if w > 1.15 * W or h > 1.15 * H:
            continue          # spans several sheets placed side by side: not this sheet's frame
        ix = max(0.0, min(r[2], W) - max(r[0], 0.0))
        iy = max(0.0, min(r[3], H) - max(r[1], 0.0))
        if ix * iy < 0.5 * W * H:
            continue
        if text_centres:
            inside = sum(1 for c in text_centres if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3])
            if inside < 0.75 * len(text_centres):
                continue
        if any(all(abs(r[k] - f[k]) <= 0.2 * U for k in range(4)) for f in frames):
            continue
        frames.append(r)
    frames.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    return {"frames": [tuple(rnd(v) for v in f) for f in frames], "inner": frames[-1] if frames else None,
            "band_tol": cfg.border_band_tol, "text_tol": 0.1 * U}


def _outside(p: Sequence[float], r: Sequence[float], tol: float) -> bool:
    return p[0] < r[0] - tol or p[0] > r[2] + tol or p[1] < r[1] - tol or p[1] > r[3] + tol


def _inside(p: Sequence[float], r: Sequence[float], tol: float) -> bool:
    return r[0] + tol < p[0] < r[2] - tol and r[1] + tol < p[1] < r[3] - tol


def is_border_segment(a: Sequence[float], b: Sequence[float], border: Optional[dict], tol: Optional[float] = None) -> bool:
    """True for a frame edge, or a line in the margin band outside the drawing area (zone ticks)."""
    if not border or not border.get("frames"):
        return False
    tol = border.get("band_tol", 1.0) if tol is None else tol
    for x0, y0, x1, y1 in border["frames"]:
        for e0, e1 in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))):
            if point_segment_distance(a, e0, e1)[0] <= tol and point_segment_distance(b, e0, e1)[0] <= tol:
                return True
    inner = border["inner"]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return _outside(mid, inner, tol) and not _inside(a, inner, tol) and not _inside(b, inner, tol)


def is_border_text(t: dict, border: Optional[dict], tol: Optional[float] = None) -> bool:
    """Zone numbers / letters printed in the margin band outside the drawing area."""
    if not (border and border.get("inner")):
        return False
    return _outside(t["center"], border["inner"], border.get("text_tol", 0.5) if tol is None else tol)


def _solid_stroke(d: dict) -> bool:
    return "s" in (d.get("type") or "") and not _is_dashed(d.get("dashes")) and not _is_white(d.get("color"))


def dominant_axis_angle(drawings: list, M, page) -> Tuple[float, float]:
    """Angle (deg) by which the drawing is rotated off the page axes, and its support.

    Length-weighted histogram of long straight solid lines, folded modulo 90 degrees.  A
    drawing whose lines are already horizontal / vertical returns 0.
    """
    min_len = 0.01 * max(page.rect.width, page.rect.height)
    hist = [0.0] * 180                          # 0.5 degree bins over [-45, 45)
    samples = []
    for d in drawings:
        if not _solid_stroke(d):
            continue
        for it in d.get("items", []):
            if it[0] != "l":
                continue
            a, b = to_view(M, it[1]), to_view(M, it[2])
            L = dist(a, b)
            if L < min_len:
                continue
            ang = ((math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + 45.0) % 90.0) - 45.0
            hist[min(179, int((ang + 45.0) * 2))] += L
            samples.append((ang, L))
    total = sum(hist)
    if total <= 0:
        return 0.0, 0.0
    sm = [hist[i - 1] + hist[i] + hist[(i + 1) % 180] for i in range(180)]
    peak = max(range(180), key=lambda i: sm[i])
    centre = peak / 2.0 - 45.0 + 0.25
    near = [(a if abs(a - centre) < 45 else a - math.copysign(90, a), L) for a, L in samples
            if min(abs(a - centre), 90 - abs(a - centre)) <= 1.0]
    theta = sum(a * L for a, L in near) / max(1e-9, sum(L for _, L in near)) if near else centre
    support = sum(L for a, L in samples if min(abs(a - theta), 90 - abs(a - theta)) <= 2.0) / total
    return theta, support


def view_transform(page, M0, theta: float):
    """Analysis frame = displayed frame rotated by -theta about the page centre.

    Returns (M, to_display, view_rect): M maps unrotated PDF coordinates into the analysis
    frame, to_display maps analysis coordinates back to the displayed page.
    """
    if abs(theta) < 1e-9:
        return M0, fitz.Matrix(1, 0, 0, 1, 0, 0), fitz.Rect(page.rect)
    cx, cy = page.rect.width / 2.0, page.rect.height / 2.0
    c, s_ = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    R = fitz.Matrix(1, 0, 0, 1, -cx, -cy) * fitz.Matrix(c, -s_, s_, c, 0, 0) * fitz.Matrix(1, 0, 0, 1, cx, cy)
    view = fitz.Rect()
    for q in (page.rect.tl, page.rect.tr, page.rect.br, page.rect.bl):
        view |= q * R
    return M0 * R, ~R, view


def estimate_text_unit(texts: List[dict]) -> float:
    """Median nominal font size of numeric labels (independent of font metrics / box padding)."""
    hs = sorted(t["font_size"] for t in texts if t["is_numeric"] and t.get("font_size", 0) > 0)
    if len(hs) < 3:
        hs = sorted(t["font_size"] for t in texts if t.get("font_size", 0) > 0)
    return hs[len(hs) // 2] if hs else REF_TEXT_UNIT


def _width_clusters(widths: Dict[float, list]) -> List[Tuple[float, list]]:
    keys = sorted(widths)
    clusters: List[List] = []
    for k in keys:
        if clusters and k - clusters[-1][0][-1] <= max(0.002, 0.06 * clusters[-1][0][-1]):
            clusters[-1][0].append(k)
            clusters[-1][1].extend(widths[k])
        else:
            clusters.append([[k], list(widths[k])])
    out = []
    for ks, segs in clusters:              # representative width: the one carrying most length
        w = max(ks, key=lambda k: sum(dist(a, b) for a, b in widths[k]))
        out.append((w, segs))
    return out


def _filled_dots(drawings: list, M, max_diameter: float) -> List[Tuple[Tuple[float, float], float]]:
    """Centres and diameters of small filled round paths (junction dot candidates)."""
    out = []
    for d in drawings:
        if "f" not in (d.get("type") or "") or d.get("fill") is None or _is_white(d.get("fill")):
            continue
        items = d.get("items", [])
        if sum(1 for it in items if it[0] == "c") < 2 or not all(it[0] in ("c", "l") for it in items):
            continue
        r = fitz.Rect(d["rect"]) * M
        if 0 < r.width <= max_diameter and r.height > 0 and abs(r.width / r.height - 1.0) < 0.2:
            out.append((((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0), (r.width + r.height) / 2.0))
    return out


def detect_wire_widths(drawings: list, M, view_rect, texts: List[dict], text_unit: float,
                       border: Optional[dict]) -> dict:
    """Harness wire stroke width(s) and the wire-number font size, chosen by evidence.

    For every stroke-width cluster of solid straight lines:
      * labels   - numeric texts sitting parallel next to its lines (within 3 of their own
                   text heights, projecting onto the line);
      * dots     - small filled dots where 3+ of its lines meet (junctions).  Frames, zone
                   ticks, tables and title blocks never meet at junction dots, so when any
                   cluster has junction dots, only clusters with dots can be the wire pen.
    The best cluster is the wire pen and the median font size of its labels is the label
    text unit (zone numbers, table entries and notes in other fonts do not count).  Further
    clusters carrying at least a quarter as many labels IN THAT FONT SIZE and a tenth of the
    long-line length are additional wire pens (e.g. bundles drawn thicker than single wires).
    """
    U = text_unit
    by_width: Dict[float, list] = defaultdict(list)
    for d in drawings:
        if not _solid_stroke(d):
            continue
        w = round(float(d.get("width") or 0.0), 3)
        if w <= 0:
            continue
        for it in d.get("items", []):
            if it[0] != "l":
                continue
            a, b = to_view(M, it[1]), to_view(M, it[2])
            if dist(a, b) < 0.3 * U or is_border_segment(a, b, border):
                continue
            if not (view_rect.x0 - U <= max(a[0], b[0]) and min(a[0], b[0]) <= view_rect.x1 + U
                    and view_rect.y0 - U <= max(a[1], b[1]) and min(a[1], b[1]) <= view_rect.y1 + U):
                continue
            by_width[w].append((a, b))
    numeric = [t for t in texts if t["is_numeric"] and t["orientation"] in ("horizontal", "vertical")
               and t.get("font_size", 0) > 0]
    dots = _filled_dots(drawings, M, 1.5 * U)
    stats = []
    for w, segs in _width_clusters(by_width):
        idx = GridIndex(30.0)
        for i, (a, b) in enumerate(segs):
            idx.insert(i, a[0], a[1], b[0], b[1])
        hits = []
        for t in numeric:
            cx, cy = t["center"]
            f = t["font_size"]
            hit = False
            for i in idx.query(cx - 3 * f, cy - 3 * f, cx + 3 * f, cy + 3 * f):
                a, b = segs[i]
                if dist(a, b) < 1.0 * f:            # a label-carrying wire piece is at least a label long
                    continue
                dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
                if t["orientation"] == "horizontal" and dy <= 0.1 * dx:
                    lo, hi, perp = min(a[0], b[0]), max(a[0], b[0]), abs(cy - (a[1] + b[1]) / 2)
                    along = cx
                elif t["orientation"] == "vertical" and dx <= 0.1 * dy:
                    lo, hi, perp = min(a[1], b[1]), max(a[1], b[1]), abs(cx - (a[0] + b[0]) / 2)
                    along = cy
                else:
                    continue
                if lo - 0.4 * f <= along <= hi + 0.4 * f and perp <= 3 * f:
                    hit = True
                    break
            if hit:
                hits.append(f)
        n_dots = 0
        for c, dia in dots:
            reach = 0.5 * dia + max(2.0 * w, 0.06 * U)
            deg = 0
            for i in idx.query(c[0] - reach, c[1] - reach, c[0] + reach, c[1] + reach):
                a, b = segs[i]
                if dist(c, a) <= reach or dist(c, b) <= reach:
                    deg += 1                        # a line ending at the dot
                elif point_segment_distance(c, a, b)[0] <= reach:
                    deg += 2                        # a line passing through the dot
                if deg >= 3:
                    n_dots += 1
                    break
        long_len = sum(dist(a, b) for a, b in segs if dist(a, b) >= 1.6 * U)
        stats.append({"width": w, "labels": len(hits), "junction_dots": n_dots, "long_length": rnd(long_len, 1),
                      "lines": len(segs), "_fonts": sorted(hits)})
    if not stats:
        return {"widths": [], "primary": None, "clusters": [], "label_text_unit": None}
    max_dots = max(c["junction_dots"] for c in stats)
    pool = [c for c in stats if c["junction_dots"] >= max(3, 0.2 * max_dots)] if max_dots >= 3 else stats
    best = max(pool, key=lambda c: (c["labels"], c["long_length"]))
    if best["labels"] < 3:
        best = max(pool, key=lambda c: c["long_length"])
    fonts = best["_fonts"]
    label_unit = fonts[len(fonts) // 2] if len(fonts) >= 5 else None
    widths = [best["width"]]
    for c in stats:
        if c is best:
            continue
        same_font = c["labels"] if label_unit is None else \
            sum(1 for f in c["_fonts"] if abs(f - label_unit) <= 0.15 * label_unit)
        if same_font >= max(5, 0.25 * best["labels"]) and c["long_length"] >= 0.1 * best["long_length"]:
            widths.append(c["width"])
    clusters = [{k: v for k, v in c.items() if k != "_fonts"} for c in sorted(stats, key=lambda c: -c["labels"])]
    return {"widths": widths, "primary": best["width"], "clusters": clusters, "label_text_unit": label_unit}


def _is_wire_width(width: float, cfg: EngineConfig) -> bool:
    ws = cfg.wire_widths or ((cfg.wire_width,) if cfg.wire_width is not None else ())
    return any(abs(width - w) <= max(min(cfg.wire_width_tol, 0.15 * w), 0.06 * w) for w in ws)


def is_significant_line(o: dict, cfg: EngineConfig) -> bool:
    """Symbol / leader / tick geometry, as opposed to hatching texture and dashed sleeves."""
    if o.get("dashed"):
        return False
    return o["width"] >= cfg.significant_min_width or dist(o["p0"], o["p1"]) >= cfg.significant_min_length


def extract_vectors(page, M, cfg: EngineConfig, drawings: Optional[list] = None,
                    border: Optional[dict] = None, view_rect=None) -> dict:
    """Extract wires, raw circles, other (non-wire) lines and closed symbol shapes.

    Sheet-border lines (frame edges, zone ticks in the margin band) are never wires and
    never terminal-symbol geometry, whatever their stroke width.  Small closed shapes
    drawn with the wire pen (terminal boxes) are symbols, not wires.
    """
    drawings = page.get_drawings() if drawings is None else drawings
    if not (cfg.wire_widths or cfg.wire_width is not None):
        raise RuntimeError("No harness wire stroke width known; pass --wire-width.")
    prect = view_rect if view_rect is not None else page.rect
    margin = cfg.layout_unit
    U = cfg.layout_unit

    def on_page(p) -> bool:
        return (prect.x0 - margin <= p[0] <= prect.x1 + margin) and (prect.y0 - margin <= p[1] <= prect.y1 + margin)

    wires: List[dict] = []
    circles: List[dict] = []
    other_lines: List[dict] = []
    shapes: List[dict] = []
    border_lines = 0
    for di, d in enumerate(drawings):
        dtype = d.get("type") or ""
        width = float(d.get("width") or 0.0)
        stroked = "s" in dtype and not _is_white(d.get("color"))
        filled = "f" in dtype and d.get("fill") is not None and not _is_white(d.get("fill"))
        if not stroked and not filled:
            continue
        dashed = _is_dashed(d.get("dashes"))
        view_items = list(_path_items_view(d.get("items", []), M))
        if not view_items:
            continue
        circ = _circle_from_items(view_items, cfg)
        if circ is not None:
            (cx, cy), diam = circ
            if on_page((cx, cy)):
                circles.append({
                    "center": (cx, cy), "diameter": diam, "filled": bool(filled),
                    "stroked": bool(stroked), "stroke_width": width, "drawing_index": di,
                })
            continue
        segs = _view_segments(view_items)
        closed = False
        if len(segs) >= 3 and dist(segs[0][0], segs[-1][1]) <= 0.06 * U:
            closed = True
        if any(k == "poly" for k, _ in view_items):
            closed = True
        is_wire = stroked and not dashed and _is_wire_width(width, cfg)
        if is_wire and closed and segs:
            xs = [p[0] for sg in segs for p in (sg[0], sg[1])]
            ys = [p[1] for sg in segs for p in (sg[0], sg[1])]
            if max(max(xs) - min(xs), max(ys) - min(ys)) <= 4.0 * U:
                is_wire = False                    # a small closed outline is a symbol (terminal box)
        for (a, b, ii, kind) in segs:
            if is_border_segment(a, b, border):
                border_lines += 1
                continue
            if is_wire:
                clipped = clip_segment(a, b, prect)   # wires running off the sheet end at the page edge
                if clipped is None:
                    continue
                a, b = clipped
            if dist(a, b) < (cfg.min_wire_length if is_wire else 0.05):
                continue
            if not (on_page(a) or on_page(b)):
                continue
            rec = {"p0": a, "p1": b, "width": width, "drawing_index": di, "item_index": ii}
            if is_wire:
                wires.append(rec)
            else:
                rec["dashed"] = dashed
                rec["filled"] = bool(filled)
                other_lines.append(rec)
        if not is_wire and closed and segs and not all(is_border_segment(a, b, border) for a, b, _, _ in segs):
            xs = [p[0] for s in segs for p in (s[0], s[1])]
            ys = [p[1] for s in segs for p in (s[0], s[1])]
            bb = (min(xs), min(ys), max(xs), max(ys))
            if (bb[2] - bb[0]) < 0.5 * prect.width and (bb[3] - bb[1]) < 0.5 * prect.height:
                shapes.append({
                    "bbox": bb, "width": width, "dashed": dashed, "filled": bool(filled),
                    "drawing_index": di, "segments": [(s[0], s[1]) for s in segs],
                })
    for i, w in enumerate(wires):
        w["id"] = f"W{i:05d}"
    for i, o in enumerate(other_lines):
        o["id"] = f"L{i:05d}"
    for i, s in enumerate(shapes):
        s["id"] = f"S{i:04d}"
    return {"wire_width": cfg.wire_width, "wire_widths": list(cfg.wire_widths or [cfg.wire_width]),
            "wires": wires, "circles_raw": circles, "other_lines": other_lines, "shapes": shapes,
            "border_lines_excluded": border_lines}


# ---------------------------------------------------------------------------
# Segmentation normalisation
# ---------------------------------------------------------------------------

def merge_collinear_wires(wires: List[dict], tol: float, angle_tol_deg: float = 1.0) -> List[dict]:
    """Merge duplicated, overlapping and touching collinear wire pieces into maximal lines.

    One wire drawn as many pieces, in any order or direction, with duplicates or small
    overlaps / gaps up to tol becomes one straight line.  Junctions, bends and T-joints are
    re-created later from dots and endpoints, so nothing structural is lost.
    """
    recs = []
    for w in wires:
        a, b = w["p0"], w["p1"]
        L = dist(a, b)
        if L <= 1e-9:
            continue
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        recs.append((ang, w))
    recs.sort(key=lambda r: r[0])
    groups: List[List[Tuple[float, dict]]] = []
    for ang, w in recs:
        if groups and (ang - groups[-1][0][0]) <= angle_tol_deg:
            groups[-1].append((ang, w))
        else:
            groups.append([(ang, w)])
    if len(groups) > 1 and (groups[0][0][0] + 180.0 - groups[-1][-1][0]) <= angle_tol_deg:
        groups[0] = groups.pop() + groups[0]
    out: List[dict] = []
    for g in groups:
        ref = g[0][1]
        u = unit((ref["p1"][0] - ref["p0"][0], ref["p1"][1] - ref["p0"][1]))
        nrm = (-u[1], u[0])
        items = []
        for _, w in g:
            c = (w["p0"][0] + w["p1"][0]) / 2 * nrm[0] + (w["p0"][1] + w["p1"][1]) / 2 * nrm[1]
            s0 = w["p0"][0] * u[0] + w["p0"][1] * u[1]
            s1 = w["p1"][0] * u[0] + w["p1"][1] * u[1]
            items.append((c, min(s0, s1), max(s0, s1), w))
        items.sort(key=lambda it: it[0])
        lanes: List[List] = []
        for it in items:
            if lanes and it[0] - lanes[-1][-1][0] <= 0.5 * tol:
                lanes[-1].append(it)
            else:
                lanes.append([it])
        for lane in lanes:
            lane.sort(key=lambda it: it[1])
            run = [lane[0]]
            for it in lane[1:]:
                if it[1] <= max(r[2] for r in run) + tol:
                    run.append(it)
                else:
                    out.append(_merged_line(run, u, nrm))
                    run = [it]
            out.append(_merged_line(run, u, nrm))
    for i, w in enumerate(out):
        w["id"] = f"W{i:05d}"
    return out


def _merged_line(run: list, u, nrm) -> dict:
    """Merged line between the run's two extreme REAL endpoints (no re-projection, so a long
    line keeps its exact geometry even when the group direction differs by a fraction of a degree)."""
    pts = [p for it in run for p in (it[3]["p0"], it[3]["p1"])]
    p0 = min(pts, key=lambda p: p[0] * u[0] + p[1] * u[1])
    p1 = max(pts, key=lambda p: p[0] * u[0] + p[1] * u[1])
    first = run[0][3]
    return {"p0": p0, "p1": p1, "width": first["width"], "drawing_index": first["drawing_index"],
            "item_index": first["item_index"], "source_pieces": len(run)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_labels(wires: List[dict], texts: List[dict], text_unit: float, axis_tol_deg: float) -> dict:
    """Label side convention per wire orientation and typical label distance, from the data.

    For every numeric text, the nearest parallel wire line it projects onto (within four
    text heights) gives a signed offset (+ = below / right).  A clear majority on one side
    fixes the convention: +1 below / right, -1 above / left, 0 printed on the wire.  The
    distance limit is 2.4 x the median offset on the convention side.
    """
    U = text_unit
    tn = math.tan(math.radians(axis_tol_deg))
    idx = GridIndex(30.0)
    for i, w in enumerate(wires):
        idx.insert(i, w["p0"][0], w["p0"][1], w["p1"][0], w["p1"][1])
    raws: Dict[str, List[float]] = {"horizontal": [], "vertical": []}
    for t in texts:
        if not t["is_numeric"] or t["orientation"] not in raws:
            continue
        cx, cy = t["center"]
        best = None
        for i in idx.query(cx - 4 * U, cy - 4 * U, cx + 4 * U, cy + 4 * U):
            a, b = wires[i]["p0"], wires[i]["p1"]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if t["orientation"] == "horizontal" and abs(dy) <= tn * abs(dx):
                lo, hi, along = min(a[0], b[0]), max(a[0], b[0]), cx
                tt = 0.0 if abs(dx) < 1e-9 else (cx - a[0]) / dx
                raw = cy - (a[1] + tt * dy)
            elif t["orientation"] == "vertical" and abs(dx) <= tn * abs(dy):
                lo, hi, along = min(a[1], b[1]), max(a[1], b[1]), cy
                tt = 0.0 if abs(dy) < 1e-9 else (cy - a[1]) / dy
                raw = cx - (a[0] + tt * dx)
            else:
                continue
            if lo - 0.4 * U <= along <= hi + 0.4 * U and abs(raw) <= 4 * U and (best is None or abs(raw) < abs(best)):
                best = raw
        if best is not None:
            raws[t["orientation"]].append(best)
    eps = 0.15 * U
    conv: Dict[str, Optional[int]] = {}
    evidence = {}
    for o, rs in raws.items():
        n = len(rs)
        pos = sum(1 for r in rs if r > eps)
        neg = sum(1 for r in rs if r < -eps)
        inl = n - pos - neg
        evidence[o] = {"samples": n, "positive": pos, "negative": neg, "on_wire": inl}
        if n < 4:
            conv[o] = None
        elif pos >= 0.65 * n:
            conv[o] = 1
        elif neg >= 0.65 * n:
            conv[o] = -1
        elif inl >= 0.5 * n:
            conv[o] = 0
        else:
            conv[o] = None
    dists = []
    for o, rs in raws.items():
        c = conv[o] if conv[o] is not None else 1
        for r in rs:
            if c == 0 and abs(r) <= eps * 3:
                dists.append(abs(r))
            elif c != 0 and c * r > eps:
                dists.append(c * r)
    dists.sort()
    median = dists[len(dists) // 2] if dists else None
    return {"side_h": conv["horizontal"], "side_v": conv["vertical"], "median_distance": median,
            "evidence": evidence}


def calibrate_endpoint_tol(wires: List[dict], base: float) -> float:
    """Endpoint join tolerance from the drawing's own gap distribution.

    Distances from every wire end to the nearest other wire form a cluster near zero
    (sloppy joins) and a far tail (free ends).  The tolerance covers that near cluster,
    never less than base and never more than 1.5 x base (joins at dots and circles are
    covered separately by snapping to the symbol, so a wide tolerance is never needed and
    would merge neighbouring junction ends).
    """
    idx = GridIndex(20.0)
    for i, w in enumerate(wires):
        idx.insert(i, w["p0"][0], w["p0"][1], w["p1"][0], w["p1"][1])
    window = 4.0 * base
    small = []
    for i, w in enumerate(wires):
        for p in (w["p0"], w["p1"]):
            best = None
            for j in idx.query(p[0] - window, p[1] - window, p[0] + window, p[1] + window):
                if j == i:
                    continue
                d, _ = point_segment_distance(p, wires[j]["p0"], wires[j]["p1"])
                if best is None or d < best:
                    best = d
            if best is not None and 1e-3 < best <= window:
                small.append(best)
    small.sort()
    if len(small) < 3:
        return base
    cut = None
    for i in range(len(small) - 1):
        if small[i + 1] > max(2.0 * small[i], small[i] + base):
            cut = small[i]
            break
    if cut is None:
        cut = small[-1] if small[-1] <= 0.75 * window else None
    if cut is None:
        return 1.5 * base if small[-1] > base else base
    return min(1.5 * base, max(base, 1.25 * cut))


def estimate_dot_diameter(symbols: List[dict], wires: List[dict], wire_unit: float, text_unit: float) -> Optional[float]:
    """Median outer diameter of filled circles where three or more wires meet."""
    idx = GridIndex(25.0)
    for i, w in enumerate(wires):
        idx.insert(i, w["p0"][0], w["p0"][1], w["p1"][0], w["p1"][1])
    ds = []
    for s in symbols:
        if not s["has_filled_circle"] or s["outer_diameter"] > 1.5 * text_unit:
            continue
        reach = s["outer_diameter"] / 2.0 + max(1.5 * wire_unit, 0.2 * s["outer_diameter"])
        dirs = symbol_directions(s, wires, idx, None, reach=reach)
        if len(dirs) >= 3:
            ds.append(s["outer_diameter"])
    ds.sort()
    return ds[len(ds) // 2] if ds else None


def layout_unit(Ut: float, Uw: float, D: Optional[float], label_distance: Optional[float]) -> float:
    """Scale of the drawing's layout from three independent cues, expressed in reference text
    units: the text size, the label placement distance and the pen style (wire pen and junction
    dot combined, so a drawing with both a fine pen and small dots counts as one style cue).
    Their median; with two cues, the text size unless the other differs by half or more."""
    style = Uw * REF_TEXT_UNIT / REF_WIRE_UNIT
    if D:
        style = math.sqrt(style * D * REF_TEXT_UNIT / REF_DOT_UNIT)
    cues = [Ut, style]
    if label_distance and label_distance > 0.3 * Ut:         # labels beside (not on) their wires
        cues.append(label_distance * REF_TEXT_UNIT / REF_LABEL_DISTANCE)
    if len(cues) == 3:
        return sorted(cues)[1]
    return Ut if abs(Ut - style) / Ut < 0.5 else min(cues)


def text_angle_tolerance(texts: List[dict]) -> float:
    """Orientation tolerance (deg) from how far texts deviate from the page axes."""
    devs = sorted(min(abs(((t["angle_deg"] + 45.0) % 90.0) - 45.0), 45.0) for t in texts)
    if not devs:
        return 12.0
    q99 = devs[min(len(devs) - 1, int(0.99 * len(devs)))]
    return max(12.0, min(25.0, q99 + 3.0))


def derive_config(base: EngineConfig, prof: dict, page_size: Tuple[float, float]) -> EngineConfig:
    """Every distance threshold rescaled from the reference drawing to this PDF's units."""
    Ut, Uw, D, UL = prof["text_unit"], prof["wire_unit"], prof["dot_unit"], prof["layout_unit"]
    t, g = Ut / REF_TEXT_UNIT, UL / REF_TEXT_UNIT
    c = replace(base)
    c.text_unit, c.wire_unit, c.dot_unit, c.layout_unit = Ut, Uw, D, UL
    c.wire_width = prof["wire_width"]
    c.wire_widths = tuple(prof["wire_widths"])
    c.wire_width_tol = max(0.02, 0.06 * Uw)
    c.min_wire_length = 0.46 * Uw
    # between hatching (2.5 % of the layout unit) and leaders (6 %) on the reference drawing, with
    # the same margin both ways; independent of the wire pen, so a heavy or fine wire pen does
    # not turn leaders into texture or hatching into leaders
    c.significant_min_width = 0.04 * UL
    c.significant_min_length = 0.6 * UL
    c.endpoint_merge_tol = prof["endpoint_tol"]
    c.split_tol = 0.67 * prof["endpoint_tol"]
    c.gap_bridge_max = 1.2 * Ut
    c.circle_merge_tol = 0.6 * g
    c.dot_max_diameter = 1.39 * D
    c.element_min_diameter = max(1.6 * D, 0.65 * UL)
    c.circle_touch_tol = max(1.5 * Uw, 0.37 * D)
    c.text_axis_tol_deg = prof["text_axis_tol_deg"]
    c.label_min_side = 0.08 * Ut
    c.label_max_side = prof["label_max_side"]
    c.label_overlap_tol = 0.35 * Ut
    c.projection_margin = 0.4 * Ut
    c.inline_max_side = prof["inline_max_side"]
    c.side_h = prof["side_h"]
    c.side_v = prof["side_v"]
    c.trace_max_length = 3.2 * max(page_size)
    c.terminal_symbol_tol = 0.79 * UL
    c.terminal_text_radius = 30.0 * g
    c.element_text_radius = 22.0 * g
    c.border_corner_tol = 0.5 * UL
    c.border_band_tol = max(2.0 * Uw, 0.2 * UL)
    del t
    return c


def calibrate_profile(page, cfg: EngineConfig) -> dict:
    """Measure the drawing: orientation, text unit, wire pen(s), dot size, label convention,
    label distance, endpoint precision and text angle tolerance.  Explicit settings in cfg
    (wire_width, label_max_side_override, side_h, side_v) are kept."""
    M0 = page.rotation_matrix
    drawings = page.get_drawings()
    theta, support = dominant_axis_angle(drawings, M0, page)
    if not cfg.deskew or abs(theta) < 0.3 or support < 0.5:
        theta = 0.0
    M, to_display, view = view_transform(page, M0, theta)
    all_texts = extract_text_objects(page, M, cfg, view)
    Ut = estimate_text_unit(all_texts)          # first guess; refined from the wire labels below
    for _ in range(2):
        pre = replace(cfg, layout_unit=Ut, text_unit=Ut, border_corner_tol=0.5 * Ut,
                      border_band_tol=max(1.0 * Ut / REF_TEXT_UNIT, 0.2 * Ut))
        border = detect_sheet_border(page, M, drawings, [tuple(t["center"]) for t in all_texts], pre)
        texts = [t for t in all_texts if not is_border_text(t, border)]
        ww = detect_wire_widths(drawings, M, view, texts, Ut, border)
        lu = ww["label_text_unit"]
        if lu is None or abs(lu - Ut) <= 0.1 * Ut:
            break
        Ut = lu                                 # e.g. zone numbers of several sheets outnumber the labels
    if cfg.wire_width is not None:
        ww = dict(ww, widths=[cfg.wire_width], primary=cfg.wire_width, explicit=True)
    elif ww["primary"] is None:
        raise RuntimeError("Could not detect the harness wire stroke width; pass --wire-width.")
    Uw = ww["primary"]
    pre = replace(pre, wire_width=Uw, wire_widths=tuple(ww["widths"]), wire_unit=Uw,
                  wire_width_tol=max(0.02, 0.06 * Uw), min_wire_length=0.46 * Uw)
    vec = extract_vectors(page, M, pre, drawings, border, view)
    base_tol = max(1.4 * Uw, 0.15 * Ut)
    wires = merge_collinear_wires(vec["wires"], base_tol)
    syms = merge_circles(vec["circles_raw"], replace(pre, circle_merge_tol=0.12 * Ut))
    D = estimate_dot_diameter(syms, wires, Uw, Ut)
    lab = calibrate_labels(wires, texts, Ut, cfg.axis_tol_deg)
    UL = layout_unit(Ut, Uw, D, lab["median_distance"])
    D = D if D else REF_DOT_UNIT * UL / REF_TEXT_UNIT
    ep_tol = calibrate_endpoint_tol(wires, max(1.4 * Uw, 0.35 * D, 0.14 * UL))
    side_h = cfg.side_h if cfg.side_h is not None else (lab["side_h"] if lab["side_h"] is not None else 1)
    side_v = cfg.side_v if cfg.side_v is not None else (lab["side_v"] if lab["side_v"] is not None else 1)
    med = lab["median_distance"]
    if cfg.label_max_side_override is not None:
        max_side = cfg.label_max_side_override
    elif med:
        max_side = max(2.4 * med, 1.2 * Ut)
    else:
        max_side = 10.5 * Ut / REF_TEXT_UNIT
    prof = {
        "deskew_deg": rnd(theta, 3), "axis_support": rnd(support, 3),
        "text_unit": rnd(Ut, 4), "wire_unit": Uw, "dot_unit": rnd(D, 4), "layout_unit": rnd(UL, 4),
        "wire_width": Uw, "wire_widths": ww["widths"], "wire_width_clusters": ww["clusters"][:6],
        "endpoint_tol": rnd(ep_tol, 4), "label_median_distance": rnd(med, 4) if med else None,
        "label_max_side": rnd(max_side, 4), "inline_max_side": rnd(max(0.6 * Ut, 1.2 * (med or 0)), 4),
        "side_h": side_h, "side_v": side_v, "side_evidence": lab["evidence"],
        "text_axis_tol_deg": rnd(text_angle_tolerance(texts), 2),
    }
    return {"profile": prof, "M": M, "to_display": to_display, "view": view, "drawings": drawings,
            "all_texts": all_texts, "texts": texts, "border": border}


# ---------------------------------------------------------------------------
# Circle symbols
# ---------------------------------------------------------------------------

def merge_circles(circles: List[dict], cfg: EngineConfig) -> List[dict]:
    """Merge duplicate / concentric circles into one physical symbol."""
    order = sorted(range(len(circles)), key=lambda i: -circles[i]["diameter"])
    groups: List[List[int]] = []
    centres: List[Tuple[float, float]] = []
    gidx = GridIndex(10.0)
    tol = cfg.circle_merge_tol
    for i in order:
        c = circles[i]["center"]
        hit = None
        for gi in sorted(gidx.query(c[0] - tol, c[1] - tol, c[0] + tol, c[1] + tol)):
            if dist(c, centres[gi]) <= tol:
                hit = gi
                break
        if hit is not None:
            groups[hit].append(i)
        else:
            groups.append([i])
            centres.append(c)
            gidx.insert(len(centres) - 1, c[0], c[1], c[0], c[1])
    symbols = []
    for members in groups:
        mem = [circles[i] for i in members]
        cx = sum(m["center"][0] for m in mem) / len(mem)
        cy = sum(m["center"][1] for m in mem) / len(mem)
        outer = max(m["diameter"] for m in mem)
        filled = [m for m in mem if m["filled"]]
        filled_d = max((m["diameter"] for m in filled), default=0.0)
        symbols.append({
            "center": (cx, cy),
            "outer_diameter": outer,
            "has_filled_circle": bool(filled),
            "filled_diameter": filled_d,
            "member_count": len(mem),
            "stroke_widths": sorted({rnd(m["stroke_width"], 3) for m in mem}),
            "drawing_indexes": sorted(m["drawing_index"] for m in mem),
        })
    symbols.sort(key=lambda s: (s["center"][1], s["center"][0]))
    for i, s in enumerate(symbols):
        s["id"] = f"C{i:04d}"
    return symbols


def cluster_directions(vectors: List[Tuple[Tuple[float, float], str]], merge_deg: float) -> List[dict]:
    """Cluster unit direction vectors into unique directions."""
    clusters: List[dict] = []
    for v, wid in vectors:
        a = angle_deg(v)
        for c in clusters:
            if angle_diff(a, c["angle_deg"]) <= merge_deg:
                c["wire_ids"].append(wid)
                break
        else:
            clusters.append({"angle_deg": a, "vector": v, "wire_ids": [wid]})
    clusters.sort(key=lambda c: c["angle_deg"])
    return clusters


def symbol_directions(sym: dict, wires: List[dict], wire_index: GridIndex, cfg: Optional[EngineConfig],
                      reach: Optional[float] = None) -> List[dict]:
    """Unique directions in which wires leave a circle symbol."""
    c = sym["center"]
    if reach is None:
        reach = sym["outer_diameter"] / 2.0 + cfg.circle_touch_tol
    vectors: List[Tuple[Tuple[float, float], str]] = []
    for wi in wire_index.query(c[0] - reach, c[1] - reach, c[0] + reach, c[1] + reach):
        w = wires[wi]
        d, _ = point_segment_distance(c, w["p0"], w["p1"])
        if d > reach:
            continue
        d0, d1 = dist(c, w["p0"]), dist(c, w["p1"])
        if d0 <= reach and d1 > reach:
            vectors.append((unit((w["p1"][0] - c[0], w["p1"][1] - c[1])), w["id"]))
        elif d1 <= reach and d0 > reach:
            vectors.append((unit((w["p0"][0] - c[0], w["p0"][1] - c[1])), w["id"]))
        elif d0 > reach and d1 > reach:
            u = unit((w["p1"][0] - w["p0"][0], w["p1"][1] - w["p0"][1]))
            vectors.append((u, w["id"]))
            vectors.append(((-u[0], -u[1]), w["id"]))
    return cluster_directions(vectors, cfg.direction_merge_deg if cfg else 12.0)


def classify_symbol(sym: dict, n_dirs: int, cfg: EngineConfig) -> str:
    d = sym["outer_diameter"]
    if d >= cfg.element_min_diameter:
        solid = sym["has_filled_circle"] and sym["filled_diameter"] >= 0.5 * d
        return "large_solid_element" if solid else "large_hollow_element"
    if not sym["has_filled_circle"]:
        return "hollow_symbol"
    if n_dirs >= 3:
        return "junction_dot"
    if n_dirs == 2:
        return "two_direction_dot"
    if n_dirs == 1:
        return "endpoint_dot"
    return "isolated_dot"


def build_symbols(vec: dict, cfg: EngineConfig) -> List[dict]:
    symbols = merge_circles(vec["circles_raw"], cfg)
    windex = GridIndex(25.0)
    for i, w in enumerate(vec["wires"]):
        windex.insert(i, w["p0"][0], w["p0"][1], w["p1"][0], w["p1"][1])
    for s in symbols:
        dirs = symbol_directions(s, vec["wires"], windex, cfg)
        s["directions"] = dirs
        s["kind"] = classify_symbol(s, len(dirs), cfg)
    return symbols


# ---------------------------------------------------------------------------
# Connectivity graph
# ---------------------------------------------------------------------------

BOUNDARY_KINDS = ("junction", "branch", "element", "terminal", "open")


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _split_wires(wires: List[dict], anchors: List[Tuple[float, float]], cfg: EngineConfig) -> List[dict]:
    """Split wire vectors at anchors (T-junction endpoints, dot centres) lying on their interior."""
    aidx = GridIndex(20.0)
    for i, a in enumerate(anchors):
        aidx.insert(i, a[0], a[1], a[0], a[1])
    pieces: List[dict] = []
    for w in wires:
        p0, p1 = w["p0"], w["p1"]
        L = dist(p0, p1)
        tol = cfg.split_tol
        ts = []
        for ai in aidx.query(min(p0[0], p1[0]) - tol, min(p0[1], p1[1]) - tol,
                             max(p0[0], p1[0]) + tol, max(p0[1], p1[1]) + tol):
            a = anchors[ai]
            d, t = point_segment_distance(a, p0, p1)
            if d <= tol and tol < t * L < L - tol:
                ts.append(t)
        ts = sorted(set(round(t, 6) for t in ts))
        pts = [p0] + [(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])) for t in ts] + [p1]
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            if dist(a, b) < 1e-6:
                continue
            pieces.append({
                "id": w["id"] if len(pts) == 2 else f"{w['id']}.{k}",
                "wire_id": w["id"], "p0": a, "p1": b,
            })
    return pieces


def build_wire_graph(vec: dict, symbols: List[dict], cfg: EngineConfig) -> dict:
    """Merged endpoints become vertices, wire vectors become edges."""
    snap_symbols = [s for s in symbols
                    if s["outer_diameter"] >= cfg.element_min_diameter or s["has_filled_circle"]]
    anchors: List[Tuple[float, float]] = []
    for w in vec["wires"]:
        anchors.append(w["p0"])
        anchors.append(w["p1"])
    for s in snap_symbols:
        anchors.append(s["center"])
    pieces = _split_wires(vec["wires"], anchors, cfg)

    sidx = GridIndex(25.0)
    for i, s in enumerate(snap_symbols):
        r = s["outer_diameter"] / 2.0 + cfg.endpoint_merge_tol
        sidx.insert(i, s["center"][0] - r, s["center"][1] - r, s["center"][0] + r, s["center"][1] + r)

    # Endpoint list: 2 per piece
    ends: List[Tuple[float, float]] = []
    for p in pieces:
        ends.append(p["p0"])
        ends.append(p["p1"])
    snapped: List[Optional[int]] = [None] * len(ends)
    for k, e in enumerate(ends):
        best, best_d = None, None
        for si in sidx.query(e[0], e[1], e[0], e[1]):
            s = snap_symbols[si]
            d = dist(e, s["center"])
            r = s["outer_diameter"] / 2.0 + cfg.endpoint_merge_tol
            if d <= r:
                score = d - s["outer_diameter"] / 2.0
                if best is None or score < best_d:
                    best, best_d = si, score
        snapped[k] = best

    uf = _UnionFind(len(ends))
    eidx = GridIndex(5.0)
    for k, e in enumerate(ends):
        if snapped[k] is None:
            eidx.insert(k, e[0], e[1], e[0], e[1])
    tol = cfg.endpoint_merge_tol
    for k, e in enumerate(ends):
        if snapped[k] is not None:
            continue
        for j in eidx.query(e[0] - tol, e[1] - tol, e[0] + tol, e[1] + tol):
            if j != k and dist(e, ends[j]) <= tol:
                uf.union(k, j)

    # Create nodes
    node_of_key: Dict[Tuple[str, int], int] = {}
    nodes: List[dict] = []
    end_node: List[int] = [0] * len(ends)
    for k, e in enumerate(ends):
        key = ("S", snapped[k]) if snapped[k] is not None else ("V", uf.find(k))
        if key not in node_of_key:
            node_of_key[key] = len(nodes)
            nodes.append({"id": len(nodes), "members": [], "symbol": None})
        n = nodes[node_of_key[key]]
        n["members"].append(e)
        if snapped[k] is not None:
            n["symbol"] = snap_symbols[snapped[k]]
        end_node[k] = node_of_key[key]
    for n in nodes:
        if n["symbol"] is not None:
            n["point"] = tuple(n["symbol"]["center"])
        else:
            xs = [m[0] for m in n["members"]]
            ys = [m[1] for m in n["members"]]
            n["point"] = (sum(xs) / len(xs), sum(ys) / len(ys))
        n["edges"] = []

    # Edges (deduplicate identical overlapping pieces)
    edges: List[dict] = []
    seen: Dict[Tuple[int, int, int], int] = {}
    for pi, p in enumerate(pieces):
        u, v = end_node[2 * pi], end_node[2 * pi + 1]
        if u == v:
            continue
        L = dist(p["p0"], p["p1"])
        key = (min(u, v), max(u, v), int(round(L)))
        if key in seen:
            edges[seen[key]]["wire_ids"].append(p["id"])
            continue
        e = {"id": len(edges), "u": u, "v": v, "pu": p["p0"], "pv": p["p1"],
             "length": L, "wire_ids": [p["id"]]}
        seen[key] = e["id"]
        edges.append(e)
        nodes[u]["edges"].append(e["id"])
        nodes[v]["edges"].append(e["id"])

    graph = {"nodes": nodes, "edges": edges, "pieces": pieces}
    bridge_gaps(graph, vec, cfg)
    drop_table_components(graph, vec.get("page_size"), vec.get("border"), cfg)
    classify_nodes(graph, vec, cfg)
    return graph


def bridge_gaps(graph: dict, vec: dict, cfg: EngineConfig) -> None:
    """Join two free wire ends that face each other on one line with nothing but text between.

    Covers wires broken by imprecise drawing and wires interrupted where a number is
    printed on the wire.  Gaps containing symbol geometry or circles are never bridged.
    """
    nodes, edges = graph["nodes"], graph["edges"]
    U = cfg.text_unit
    free = []
    for n in nodes:
        if n["symbol"] is None and len(n["edges"]) == 1:
            _, near, far = edge_far(graph, n["edges"][0], n["id"])
            free.append((n["id"], unit((near[0] - far[0], near[1] - far[1]))))
    if len(free) < 2:
        graph["bridged_gaps"] = 0
        return
    oidx = GridIndex(20.0)
    sig = [o for o in vec["other_lines"] if is_significant_line(o, cfg)]
    for i, o in enumerate(sig):
        oidx.insert(i, o["p0"][0], o["p0"][1], o["p1"][0], o["p1"][1])
    tidx = GridIndex(20.0)
    texts = vec.get("texts") or []
    for i, t in enumerate(texts):
        tidx.insert(i, *t["bbox"])
    fidx = GridIndex(20.0)
    for k, (nid, _) in enumerate(free):
        p = nodes[nid]["point"]
        fidx.insert(k, p[0], p[1], p[0], p[1])
    cands = []
    reach = max(cfg.gap_bridge_max, 4.0 * U)
    cos_tol = math.cos(math.radians(5.0))
    for k, (a, ua) in enumerate(free):
        pa = nodes[a]["point"]
        for j in fidx.query(pa[0] - reach, pa[1] - reach, pa[0] + reach, pa[1] + reach):
            if j <= k:
                continue
            b, ub = free[j]
            pb = nodes[b]["point"]
            gap = dist(pa, pb)
            if gap <= cfg.endpoint_merge_tol or gap > reach:
                continue
            v = unit((pb[0] - pa[0], pb[1] - pa[1]))
            if ua[0] * v[0] + ua[1] * v[1] < cos_tol or -(ub[0] * v[0] + ub[1] * v[1]) < cos_tol:
                continue
            limit = cfg.gap_bridge_max
            for ti in tidx.query(min(pa[0], pb[0]) - U, min(pa[1], pb[1]) - U, max(pa[0], pb[0]) + U, max(pa[1], pb[1]) + U):
                t = texts[ti]
                d, tp = point_segment_distance(t["center"], pa, pb)
                if t["is_numeric"] and d <= 0.6 * U and 0.0 < tp < 1.0:
                    ext = max(t["bbox"][2] - t["bbox"][0], t["bbox"][3] - t["bbox"][1])
                    limit = max(limit, ext + 1.0 * U)       # wire interrupted by a printed number
            if gap > limit:
                continue
            blocked = False
            for oi in oidx.query(min(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[0], pb[0]), max(pa[1], pb[1])):
                o = sig[oi]
                if point_segment_distance(o["p0"], pa, pb)[0] <= 0.3 * U or \
                        point_segment_distance(o["p1"], pa, pb)[0] <= 0.3 * U:
                    blocked = True
                    break
            if not blocked:
                for c in vec.get("circles_raw", []):
                    if point_segment_distance(c["center"], pa, pb)[0] <= c["diameter"] / 2 + 0.2 * U:
                        blocked = True
                        break
            if not blocked:
                cands.append((gap, a, b))
    cands.sort()
    used = set()
    n_br = 0
    for gap, a, b in cands:
        if a in used or b in used:
            continue
        used |= {a, b}
        e = {"id": len(edges), "u": a, "v": b, "pu": nodes[a]["point"], "pv": nodes[b]["point"],
             "length": gap, "wire_ids": ["BRIDGE"]}
        edges.append(e)
        nodes[a]["edges"].append(e["id"])
        nodes[b]["edges"].append(e["id"])
        n_br += 1
    graph["bridged_gaps"] = n_br


def drop_table_components(graph: dict, page_size: Optional[Tuple[float, float]], border: Optional[dict],
                          cfg: Optional[EngineConfig] = None) -> None:
    """Remove wire-width components that are drawing tables or frames, not harness.

    A component is dropped when it has no junction dot or circular element, only
    horizontal / vertical edges, and either closed cells (two or more independent cycles,
    or one cycle when it touches the drawing frame or spans most of the page, e.g. an
    undetected frame), or every free end resting on the drawing frame (a title block or
    revision table hanging on the frame; table lines cross without connecting, so such a
    table can be cycle-free).  Harness routes have dotted junctions and end at terminals.
    """
    frame_edges = []
    for x0, y0, x1, y1 in (border or {}).get("frames", []):
        frame_edges += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1))]

    U = cfg.layout_unit if cfg else REF_TEXT_UNIT
    ax = 0.1 * U

    def on_frame(p) -> bool:
        return any(point_segment_distance(p, a, b)[0] <= 0.3 * U for a, b in frame_edges)

    nodes, edges = graph["nodes"], graph["edges"]
    seen = set()
    drop_edges = set()
    dropped = []
    for start in range(len(nodes)):
        if start in seen or not nodes[start]["edges"]:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            for eid in nodes[n]["edges"]:
                e = edges[eid]
                stack.extend((e["u"], e["v"]))
        seen |= comp
        ceids = {eid for n in comp for eid in nodes[n]["edges"]}
        if any(nodes[n]["symbol"] is not None for n in comp):
            continue
        if not all(abs(edges[e]["pu"][0] - edges[e]["pv"][0]) <= ax or abs(edges[e]["pu"][1] - edges[e]["pv"][1]) <= ax
                   for e in ceids):
            continue
        cycles = len(ceids) - len(comp) + 1
        xs = [p[0] for e in ceids for p in (edges[e]["pu"], edges[e]["pv"])]
        ys = [p[1] for e in ceids for p in (edges[e]["pu"], edges[e]["pv"])]
        bb = (min(xs), min(ys), max(xs), max(ys))
        big = bool(page_size) and (bb[2] - bb[0]) >= 0.6 * page_size[0] and (bb[3] - bb[1]) >= 0.6 * page_size[1]
        touches = False
        if border and border.get("inner"):
            r = border["inner"]
            touches = min(abs(bb[0] - r[0]), abs(bb[2] - r[2]), abs(bb[1] - r[1]), abs(bb[3] - r[3])) <= 0.5 * U
        free_ends = [n for n in comp if len(nodes[n]["edges"]) == 1]
        hangs = bool(frame_edges) and bool(free_ends) and all(on_frame(nodes[n]["point"]) for n in free_ends)
        if cycles >= 2 or (cycles >= 1 and (touches or big)) or hangs:
            drop_edges |= ceids
            dropped.append({"bbox": [rnd(v) for v in bb], "edges": len(ceids), "cycles": cycles,
                            "reason": "hangs on the drawing frame" if hangs and cycles < 1 else "closed table / frame cells"})
    if not drop_edges:
        graph["dropped_components"] = []
        return
    keep = [e for e in edges if e["id"] not in drop_edges]
    remap = {e["id"]: i for i, e in enumerate(keep)}
    for i, e in enumerate(keep):
        e["id"] = i
    for n in nodes:
        n["edges"] = [remap[eid] for eid in n["edges"] if eid in remap]
    graph["edges"] = keep
    graph["dropped_components"] = dropped


def edge_far(graph: dict, eid: int, node_id: int) -> Tuple[int, Tuple[float, float], Tuple[float, float]]:
    """For edge eid seen from node_id: (far node, near point, far point)."""
    e = graph["edges"][eid]
    if e["u"] == node_id:
        return e["v"], e["pu"], e["pv"]
    return e["u"], e["pv"], e["pu"]


def node_directions(graph: dict, node_id: int, cfg: EngineConfig) -> List[dict]:
    n = graph["nodes"][node_id]
    vecs = []
    for eid in n["edges"]:
        _, near, far = edge_far(graph, eid, node_id)
        vecs.append((unit((far[0] - near[0], far[1] - near[1])), str(eid)))
    return cluster_directions(vecs, cfg.direction_merge_deg)


def _near_symbol_geometry(p: Sequence[float], vec: dict, idx: GridIndex, cfg: EngineConfig) -> bool:
    tol = cfg.terminal_symbol_tol
    for li in idx.query(p[0] - tol, p[1] - tol, p[0] + tol, p[1] + tol):
        o = vec["other_lines"][li]
        if not (is_significant_line(o, cfg) or o.get("filled")):
            continue
        d, _ = point_segment_distance(p, o["p0"], o["p1"])
        if d <= tol:
            return True
    return False


def classify_nodes(graph: dict, vec: dict, cfg: EngineConfig) -> None:
    """Assign node kinds: junction, branch, element, terminal, open, pass, isolated."""
    oidx = GridIndex(20.0)
    for i, o in enumerate(vec["other_lines"]):
        oidx.insert(i, o["p0"][0], o["p0"][1], o["p1"][0], o["p1"][1])
    for n in graph["nodes"]:
        dirs = node_directions(graph, n["id"], cfg)
        n["directions"] = dirs
        nd = len(dirs)
        deg = len(n["edges"])
        sym = n["symbol"]
        n["symbol_id"] = sym["id"] if sym else None
        n["symbol_kind"] = None
        if sym is not None:
            large = sym["outer_diameter"] >= cfg.element_min_diameter
            n["symbol_kind"] = classify_symbol(sym, nd, cfg)
            if large:
                n["kind"] = "terminal" if nd == 1 else ("element" if nd >= 2 else "isolated")
                continue
            if sym["has_filled_circle"]:
                if nd >= 3:
                    n["kind"] = "junction"
                elif nd == 2:
                    n["kind"] = "pass"
                elif nd == 1:
                    n["kind"] = "terminal" if _near_symbol_geometry(n["point"], vec, oidx, cfg) else "open"
                else:
                    n["kind"] = "isolated"
                continue
        if deg == 0:
            n["kind"] = "isolated"
        elif deg == 1 or nd == 1:
            n["kind"] = "terminal" if _near_symbol_geometry(n["point"], vec, oidx, cfg) else "open"
        elif deg == 2:
            n["kind"] = "pass"
        else:
            n["kind"] = "branch"


# ---------------------------------------------------------------------------
# Strict wire-label placement rules
# ---------------------------------------------------------------------------

def build_text_index(texts: List[dict], numeric_only: bool = True) -> GridIndex:
    idx = GridIndex(20.0)
    for i, t in enumerate(texts):
        if numeric_only and not t["is_numeric"]:
            continue
        b = t["bbox"]
        idx.insert(i, b[0], b[1], b[2], b[3])
    return idx


def strict_label_metrics(p0: Sequence[float], p1: Sequence[float], text: dict,
                         cfg: EngineConfig, orientation: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """Check a numeric text against one straight wire piece.

    Returns (metrics, "ok") for a valid candidate, or (None, reason).  Analysis-frame y grows
    downwards.  The required side comes from the calibrated convention: side_h +1 = below a
    horizontal wire (the standard), -1 = above, 0 = printed on the wire; side_v +1 = right of
    a vertical wire (the standard), -1 = left, 0 = on the wire.
    """
    if not text.get("is_numeric"):
        return None, "not_numeric"
    orient = orientation or piece_orientation(p0, p1, cfg.axis_tol_deg)
    if orient not in ("horizontal", "vertical"):
        return None, "angled_piece"
    cx, cy = text["center"]
    x0, y0, x1, y1 = text["bbox"]
    margin = cfg.projection_margin
    if orient == "horizontal":
        if text["orientation"] != "horizontal":
            return None, "text_not_horizontal"
        a, b = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
        lo, hi = a[0], b[0]
        if cx < lo - margin or cx > hi + margin:
            return None, "outside_projection"
        t = 0.0 if hi - lo <= 1e-9 else min(1.0, max(0.0, (cx - lo) / (hi - lo)))
        wire_c = a[1] + t * (b[1] - a[1])
        raw = cy - wire_c                       # positive = below
        conv = 1 if cfg.side_h is None else cfg.side_h
        near_edge_gap = (y0 - wire_c) if conv >= 0 else (wire_c - y1)
        along = cx
    else:
        if text["orientation"] != "vertical":
            return None, "text_not_vertical"
        a, b = (p0, p1) if p0[1] <= p1[1] else (p1, p0)
        lo, hi = a[1], b[1]
        if cy < lo - margin or cy > hi + margin:
            return None, "outside_projection"
        t = 0.0 if hi - lo <= 1e-9 else min(1.0, max(0.0, (cy - lo) / (hi - lo)))
        wire_c = a[0] + t * (b[0] - a[0])
        raw = cx - wire_c                       # positive = right
        conv = 1 if cfg.side_v is None else cfg.side_v
        near_edge_gap = (x0 - wire_c) if conv >= 0 else (wire_c - x1)
        along = cy
    if conv == 0:                               # numbers printed on the wire
        side = abs(raw)
        limit = cfg.inline_max_side
        if side > limit:
            return None, "too_far"
    else:
        side = conv * raw
        limit = cfg.label_max_side
        if side <= cfg.label_min_side:
            return None, "wrong_side"
        if side > limit:
            return None, "too_far"
        if near_edge_gap < -cfg.label_overlap_tol:
            return None, "overlaps_wire"
    length = hi - lo
    mid = (lo + hi) / 2.0
    half = max(length / 2.0, 0.2 * cfg.text_unit)
    centre_err = abs(along - mid)
    projection = 0.5 if length <= 1e-9 else (along - lo) / length
    score = side / limit + cfg.centre_weight * min(centre_err / half, 2.0)
    return {
        "orientation": orient,
        "perpendicular_distance": rnd(side),
        "edge_gap": rnd(near_edge_gap),
        "projection": rnd(projection, 4),
        "centre_error": rnd(centre_err),
        "piece_length": rnd(length),
        "score": rnd(score, 4),
    }, "ok"


def labels_on_piece(p0, p1, texts: List[dict], tindex: GridIndex, cfg: EngineConfig) -> List[dict]:
    """All valid strict label candidates for the oriented piece p0 -> p1."""
    m = cfg.label_max_side + cfg.projection_margin + 2.0
    x0, x1 = min(p0[0], p1[0]) - m, max(p0[0], p1[0]) + m
    y0, y1 = min(p0[1], p1[1]) - m, max(p0[1], p1[1]) + m
    L = dist(p0, p1)
    out = []
    for ti in tindex.query(x0, y0, x1, y1):
        t = texts[ti]
        met, _ = strict_label_metrics(p0, p1, t, cfg)
        if met is None:
            continue
        _, tpar = point_segment_distance(t["center"], p0, p1)
        met = dict(met)
        met["text_index"] = ti
        met["along"] = tpar * L
        out.append(met)
    return out


# ---------------------------------------------------------------------------
# Depth-1 junction identity
# ---------------------------------------------------------------------------

def trace_direction(graph: dict, start_node: int, first_edge: int, texts: List[dict],
                    tindex: GridIndex, cfg: EngineConfig) -> dict:
    """Trace one outgoing direction until the first valid numeric label or a stop condition."""
    path: List[Tuple[float, float]] = []
    edge_ids: List[int] = []
    cur, eid = start_node, first_edge
    total = 0.0
    steps = 0
    visited = set()
    result = {"label": None, "label_text_id": None, "confidence": 0.0, "ambiguous": False,
              "stop_reason": None, "label_metrics": None}
    while True:
        steps += 1
        visited.add(eid)
        far, near_pt, far_pt = edge_far(graph, eid, cur)
        if not path:
            path.append(near_pt)
        path.append(far_pt)
        edge_ids.append(eid)
        cands = labels_on_piece(near_pt, far_pt, texts, tindex, cfg)
        if cands:
            cands.sort(key=lambda c: (c["along"], c["score"]))
            best = cands[0]
            close = [c for c in cands[1:] if abs(c["along"] - best["along"]) < 0.6 * cfg.layout_unit
                     and abs(c["score"] - best["score"]) < cfg.ambiguity_score_gap
                     and texts[c["text_index"]]["norm"] != texts[best["text_index"]]["norm"]]
            t = texts[best["text_index"]]
            result.update({
                "label": t["norm"], "label_text_id": t["id"], "ambiguous": bool(close),
                "stop_reason": "label_found",
                "confidence": rnd(max(0.05, 1.0 - 0.5 * best["score"]) * (0.6 if close else 1.0), 3),
                "label_metrics": {k: v for k, v in best.items() if k != "text_index"},
            })
            break
        total += dist(near_pt, far_pt)
        kind = graph["nodes"][far]["kind"]
        if kind != "pass":
            result["stop_reason"] = {
                "junction": "reached_junction", "branch": "reached_branch",
                "element": "reached_circular_element", "terminal": "wire_end_terminal",
                "open": "wire_end_open", "isolated": "wire_end",
            }.get(kind, "reached_" + kind)
            break
        nxt = [e for e in graph["nodes"][far]["edges"] if e != eid]
        if not nxt:
            result["stop_reason"] = "wire_end"
            break
        if nxt[0] in visited:
            result["stop_reason"] = "loop"
            break
        if total > cfg.trace_max_length or steps >= cfg.trace_max_steps:
            result["stop_reason"] = "trace_limit"
            break
        cur, eid = far, nxt[0]
    result["path"] = [rpt(p) for p in path]
    result["edge_ids"] = edge_ids
    result["end_node"] = far
    return result


def depth1_id(prefix: str, labels: Iterable[Optional[str]]) -> str:
    vals = [l for l in labels if l]
    n_unl = sum(1 for l in labels if not l)
    parts = sorted(vals, key=numeric_sort_key) + ["UNLABELLED"] * n_unl
    return f"{prefix}[{'|'.join(parts)}]"


def compute_depth1_ids(graph: dict, texts: List[dict], cfg: EngineConfig) -> List[dict]:
    """Depth-1 identity for every junction (and unexpected branch point)."""
    tindex = build_text_index(texts, numeric_only=True)
    out = []
    for n in graph["nodes"]:
        if n["kind"] not in ("junction", "branch"):
            continue
        outgoing = []
        for eid in n["edges"]:
            _, near, far = edge_far(graph, eid, n["id"])
            outgoing.append((angle_deg(unit((far[0] - near[0], far[1] - near[1]))), eid))
        dirs = []
        for k, (ang, eid) in enumerate(sorted(outgoing)):
            tr = trace_direction(graph, n["id"], eid, texts, tindex, cfg)
            tr["direction_index"] = k
            tr["first_edge"] = eid
            tr["angle_deg"] = rnd(ang, 2)
            dirs.append(tr)
        prefix = "DOT1" if n["kind"] == "junction" else "BRANCH1"
        sid = depth1_id(prefix, [d["label"] for d in dirs])
        n["structural_id"] = sid
        n["depth1_labels"] = [d["label"] or "UNLABELLED" for d in dirs]
        out.append({
            "node_id": n["id"],
            "kind": n["kind"],
            "symbol_id": n.get("symbol_id"),
            "point": rpt(n["point"]),
            "structural_id": sid,
            "labels": sorted([d["label"] for d in dirs if d["label"]], key=numeric_sort_key),
            "unlabelled_count": sum(1 for d in dirs if not d["label"]),
            "directions": dirs,
        })
    return out


# ---------------------------------------------------------------------------
# Whole-page extraction convenience wrapper
# ---------------------------------------------------------------------------

def extract_page(page, cfg: EngineConfig) -> dict:
    """Calibrate the drawing, then run vector extraction, symbols, graph and depth-1 ids.

    All geometry is in the analysis frame (displayed page, de-skewed when the drawing is
    rotated off the page axes); ext["to_display"] maps it back to the displayed page.
    """
    cal = calibrate_profile(page, cfg)
    prof = cal["profile"]
    M, border = cal["M"], cal["border"]
    dcfg = derive_config(cfg, prof, (page.rect.width, page.rect.height))
    tol = math.sin(math.radians(dcfg.text_axis_tol_deg))
    texts = cal["texts"]
    for i, t in enumerate(texts):
        t["id"] = f"T{i:05d}"
        dx, dy = t["dir"]
        t["orientation"] = "horizontal" if abs(dy) <= tol else ("vertical" if abs(dx) <= tol else "angled")
    vec = extract_vectors(page, M, dcfg, cal["drawings"], border, cal["view"])
    vec["wires"] = merge_collinear_wires(vec["wires"], dcfg.endpoint_merge_tol)
    vec["border"] = border
    vec["texts"] = texts
    vec["page_size"] = (page.rect.width, page.rect.height)
    symbols = build_symbols(vec, dcfg)
    graph = build_wire_graph(vec, symbols, dcfg)
    junctions = compute_depth1_ids(graph, texts, dcfg)
    border_report = {"frames": [list(f) for f in border["frames"]],
                     "drawing_area": [rnd(v) for v in border["inner"]] if border["inner"] else None,
                     "border_texts_excluded": len(cal["all_texts"]) - len(texts),
                     "border_lines_excluded": vec["border_lines_excluded"],
                     "table_components_dropped": graph.get("dropped_components", [])}
    T = cal["to_display"]
    return {"matrix": M, "to_display": T, "texts": texts, "vectors": vec, "symbols": symbols,
            "graph": graph, "junctions": junctions, "border": border_report, "profile": prof, "cfg": dcfg}


def to_display_pt(ext: dict, p: Sequence[float]) -> Tuple[float, float]:
    """Analysis-frame point -> displayed page point (identity unless the drawing was de-skewed)."""
    T = ext.get("to_display")
    if T is None:
        return (float(p[0]), float(p[1]))
    return (T.a * p[0] + T.c * p[1] + T.e, T.b * p[0] + T.d * p[1] + T.f)


def symbols_to_json(symbols: List[dict]) -> List[dict]:
    out = []
    for s in symbols:
        out.append({
            "id": s["id"], "kind": s.get("kind"), "center": rpt(s["center"]),
            "outer_diameter": rnd(s["outer_diameter"]), "has_filled_circle": s["has_filled_circle"],
            "filled_diameter": rnd(s["filled_diameter"]), "member_count": s["member_count"],
            "stroke_widths": s["stroke_widths"],
            "directions": [{"angle_deg": rnd(d["angle_deg"], 2), "vector": rpt(d["vector"], 4),
                            "wire_ids": d["wire_ids"]} for d in s.get("directions", [])],
        })
    return out


def wires_to_json(wires: List[dict]) -> List[dict]:
    return [{"id": w["id"], "p0": rpt(w["p0"]), "p1": rpt(w["p1"]), "width": rnd(w["width"], 3),
             "drawing_index": w["drawing_index"], "item_index": w["item_index"]} for w in wires]


# ---------------------------------------------------------------------------
# Diagnostic drawing helpers
# ---------------------------------------------------------------------------

def zp(p: Sequence[float], zoom: float) -> Tuple[int, int]:
    return (int(round(p[0] * zoom)), int(round(p[1] * zoom)))


def fade(img: np.ndarray, keep: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, keep, np.full_like(img, 255), 1.0 - keep, 0)


def put_label(img: np.ndarray, text: str, org: Tuple[int, int], color, scale: float = 0.45,
              thickness: int = 1, bg=(255, 255, 255)) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    x = max(0, min(x, img.shape[1] - w - 2))
    y = max(h + 2, min(y, img.shape[0] - 2))
    if bg is not None:
        cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def add_legend_strip(img: np.ndarray, entries: List[Tuple[str, Optional[tuple]]], title: Optional[str] = None) -> np.ndarray:
    """Legend in a white strip BELOW the page image, so it never hides drawing content.

    The drawing keeps its pixel mapping (point * zoom); the strip only extends the canvas.
    """
    rows = ([(title, None)] if title else []) + list(entries)
    h = 16 + 26 * len(rows)
    strip = np.full((h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.line(strip, (0, 1), (img.shape[1] - 1, 1), (120, 120, 120), 1)
    for k, (text, col) in enumerate(rows):
        y = 24 + 26 * k
        x = 12
        if col is not None:
            cv2.line(strip, (x, y - 6), (x + 34, y - 6), col, 5)
            x += 46
        cv2.putText(strip, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if col is None and k == 0 and title else 0.52,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


SYMBOL_COLORS = {
    "junction_dot": (0, 0, 230),
    "two_direction_dot": (0, 140, 255),
    "endpoint_dot": (0, 200, 220),
    "isolated_dot": (128, 128, 128),
    "hollow_symbol": (200, 160, 0),
    "large_hollow_element": (180, 0, 180),
    "large_solid_element": (120, 0, 120),
}

DIRECTION_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                    (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40)]


def draw_wires_and_text(img: np.ndarray, ext: dict, zoom: float) -> np.ndarray:
    out = fade(img)

    def zp(p, z):                          # analysis frame -> displayed pixels
        q = to_display_pt(ext, p)
        return (int(round(q[0] * z)), int(round(q[1] * z)))
    for w in ext["vectors"]["wires"]:
        cv2.line(out, zp(w["p0"], zoom), zp(w["p1"], zoom), (255, 120, 0), max(2, int(zoom)), cv2.LINE_AA)
    colors = {"horizontal": (0, 160, 0), "vertical": (200, 0, 200), "angled": (0, 120, 255)}
    for t in ext["texts"]:
        b = t["bbox"]
        col = colors[t["orientation"]] if t["is_numeric"] else (150, 150, 150)
        corners = np.array([zp(c, zoom) for c in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))], np.int32)
        cv2.polylines(out, [corners], True, col, 2 if t["is_numeric"] else 1)
    for n in ext["graph"]["nodes"]:
        if n["kind"] in ("terminal", "open", "branch", "element"):
            col = {"terminal": (0, 140, 0), "open": (0, 0, 255), "branch": (255, 0, 255),
                   "element": (180, 0, 180)}[n["kind"]]
            cv2.circle(out, zp(n["point"], zoom), max(5, int(2.5 * zoom)), col, 2, cv2.LINE_AA)
    for f in ext.get("border", {}).get("frames", []):
        corners = np.array([zp(c, zoom) for c in ((f[0], f[1]), (f[2], f[1]), (f[2], f[3]), (f[0], f[3]))], np.int32)
        cv2.polylines(out, [corners], True, (0, 200, 255), 1)
    return add_legend_strip(out, [
        ("wires", (255, 120, 0)),
        ("numeric text: horizontal", (0, 160, 0)), ("numeric text: vertical", (200, 0, 200)),
        ("numeric text: angled", (0, 120, 255)), ("non-numeric text", (150, 150, 150)),
        ("nodes: terminal", (0, 140, 0)), ("nodes: open end", (0, 0, 255)), ("nodes: branch", (255, 0, 255)),
        ("nodes: circular element", (180, 0, 180)), ("detected sheet frame (excluded)", (0, 200, 255))])


def draw_junction_diagnostics(img: np.ndarray, ext: dict, zoom: float) -> np.ndarray:
    out = fade(img)

    def zp(p, z):                          # analysis frame -> displayed pixels
        q = to_display_pt(ext, p)
        return (int(round(q[0] * z)), int(round(q[1] * z)))
    for s in ext["symbols"]:
        col = SYMBOL_COLORS.get(s.get("kind"), (0, 0, 0))
        r = max(4, int(round(s["outer_diameter"] / 2.0 * zoom)) + 3)
        cv2.circle(out, zp(s["center"], zoom), r, col, 2, cv2.LINE_AA)
    for j in ext["junctions"]:
        for k, d in enumerate(j["directions"]):
            col = DIRECTION_COLORS[k % len(DIRECTION_COLORS)]
            pts = np.array([zp(p, zoom) for p in d["path"]], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(out, [pts], False, col, max(2, int(zoom)), cv2.LINE_AA)
            if d.get("label_metrics") and d.get("label_text_id"):
                t = next((t for t in ext["texts"] if t["id"] == d["label_text_id"]), None)
                if t:
                    b = t["bbox"]
                    corners = np.array([zp(c, zoom) for c in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))],
                                       np.int32)
                    cv2.polylines(out, [corners], True, col, 2)
        jp = zp(j["point"], zoom)
        cv2.circle(out, jp, max(5, int(2 * zoom)), (0, 0, 230), -1, cv2.LINE_AA)
        put_label(out, j["structural_id"], (jp[0] + 6, jp[1] - 6), (0, 0, 160), 0.45)
    return add_legend_strip(out, [
        ("junction dot", SYMBOL_COLORS["junction_dot"]), ("two-direction dot", SYMBOL_COLORS["two_direction_dot"]),
        ("endpoint dot", SYMBOL_COLORS["endpoint_dot"]), ("hollow symbol", SYMBOL_COLORS["hollow_symbol"]),
        ("large circular element", SYMBOL_COLORS["large_hollow_element"]),
        ("coloured paths: traced outgoing directions, boxes: chosen first labels", None)])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

SIDE_CHOICES = {"auto": None, "below": 1, "above": -1, "right": 1, "left": -1, "on-wire": 0}


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--wire-width", default="auto",
                    help="Harness wire stroke width in PDF points, or 'auto' (default: measured per PDF).")
    ap.add_argument("--label-max-side", default="auto",
                    help="Maximum label distance from its wire in pt, or 'auto' (default: 2.1 x measured median).")
    ap.add_argument("--label-side-h", default="auto", choices=["auto", "below", "above", "on-wire"],
                    help="Side of horizontal wires carrying their numbers (default: detected per PDF).")
    ap.add_argument("--label-side-v", default="auto", choices=["auto", "right", "left", "on-wire"],
                    help="Side of vertical wires carrying their numbers (default: detected per PDF).")
    ap.add_argument("--no-deskew", action="store_true", help="Do not straighten a drawing rotated off the page axes.")


def engine_config_from_args(args) -> EngineConfig:
    cfg = EngineConfig()
    ww = str(args.wire_width).strip().lower()
    try:
        cfg.wire_width = None if ww == "auto" else float(ww)
    except ValueError:
        raise ValueError(f"--wire-width must be a number or 'auto' (got {args.wire_width!r})")
    lm = str(args.label_max_side).strip().lower()
    try:
        cfg.label_max_side_override = None if lm == "auto" else float(lm)
    except ValueError:
        raise ValueError(f"--label-max-side must be a number or 'auto' (got {args.label_max_side!r})")
    cfg.side_h = SIDE_CHOICES[args.label_side_h]
    cfg.side_v = SIDE_CHOICES[args.label_side_v]
    cfg.deskew = not args.no_deskew
    return cfg


def run(pdf: str, out_dir: str, page_no: int, zoom: float, cfg: EngineConfig, images: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    doc = None
    try:
        doc, page = open_pdf_page(pdf, page_no)
        ext = extract_page(page, cfg)
        stem = os.path.splitext(os.path.basename(pdf))[0]
        kinds: Dict[str, int] = defaultdict(int)
        for s in ext["symbols"]:
            kinds[s["kind"]] += 1
        node_kinds: Dict[str, int] = defaultdict(int)
        for n in ext["graph"]["nodes"]:
            node_kinds[n["kind"]] += 1
        data = {
            "schema": "harness-junctions/1",
            "generator": {"script": os.path.basename(__file__), "engine": ENGINE_NAME, "version": ENGINE_VERSION},
            "coordinate_frame": "analysis_frame" if ext["profile"]["deskew_deg"] else "displayed_page",
            "analysis_to_display": [rnd(v, 6) for v in ext["to_display"]],
            "profile": ext["profile"],
            "source_pdf": source_pdf_info(pdf, page),
            "config": ext["cfg"].to_dict(),
            "wire_width_used": ext["vectors"]["wire_width"],
            "summary": {
                "wire_vectors": len(ext["vectors"]["wires"]),
                "texts": len(ext["texts"]),
                "numeric_texts": sum(1 for t in ext["texts"] if t["is_numeric"]),
                "raw_circles": len(ext["vectors"]["circles_raw"]),
                "circle_symbols": len(ext["symbols"]),
                "symbol_kinds": dict(kinds),
                "graph_nodes": len(ext["graph"]["nodes"]),
                "graph_edges": len(ext["graph"]["edges"]),
                "node_kinds": dict(node_kinds),
                "junctions": sum(1 for j in ext["junctions"] if j["kind"] == "junction"),
                "branch_points": sum(1 for j in ext["junctions"] if j["kind"] == "branch"),
                "sheet_frames": len(ext["border"]["frames"]),
                "border_texts_excluded": ext["border"]["border_texts_excluded"],
                "border_lines_excluded": ext["border"]["border_lines_excluded"],
                "table_components_dropped": len(ext["border"]["table_components_dropped"]),
            },
            "sheet_border": ext["border"],
            "junctions": ext["junctions"],
            "symbols": symbols_to_json(ext["symbols"]),
            "texts": ext["texts"],
            "wires": wires_to_json(ext["vectors"]["wires"]),
        }
        json_path = os.path.join(out_dir, f"{stem}_p{page_no}_junctions.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        written = [json_path]
        if images:
            base = render_page_bgr(page, zoom)
            p1 = os.path.join(out_dir, f"{stem}_p{page_no}_wires_text.png")
            p2 = os.path.join(out_dir, f"{stem}_p{page_no}_junctions.png")
            cv2.imwrite(p1, draw_wires_and_text(base, ext, zoom))
            cv2.imwrite(p2, draw_junction_diagnostics(base, ext, zoom))
            written += [p1, p2]
        data["_written"] = written
        return data
    finally:
        if doc is not None:
            doc.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Script 1: junction dots and depth-1 junction identities.")
    ap.add_argument("--pdf", required=True, help="Input harness PDF.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--page", type=int, default=0, help="Zero-based page number (default 0).")
    ap.add_argument("--zoom", type=float, default=3.0, help="Diagnostic rendering zoom (default 3.0).")
    ap.add_argument("--no-images", action="store_true", help="Skip diagnostic images.")
    add_engine_args(ap)
    args = ap.parse_args(argv)
    if args.zoom <= 0:
        ap.error("--zoom must be positive")
    try:
        cfg = engine_config_from_args(args)
        data = run(args.pdf, args.out, args.page, args.zoom, cfg, images=not args.no_images)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    s = data["summary"]
    print(f"PDF            : {args.pdf} (page {args.page})")
    pr = data["profile"]
    print(f"Wire width     : {data['wire_width_used']}  (wire pens {pr['wire_widths']})")
    print(f"Calibration    : text unit {pr['text_unit']}, dot {pr['dot_unit']}, layout unit {pr['layout_unit']}, "
          f"de-skew {pr['deskew_deg']} deg, label side h={pr['side_h']} v={pr['side_v']}, "
          f"label max {pr['label_max_side']}, endpoint tol {pr['endpoint_tol']}")
    print(f"Wire vectors   : {s['wire_vectors']}   texts: {s['texts']} (numeric {s['numeric_texts']})")
    print(f"Circle symbols : {s['circle_symbols']}  {s['symbol_kinds']}")
    print(f"Graph          : {s['graph_nodes']} nodes, {s['graph_edges']} edges  {s['node_kinds']}")
    print(f"Junctions      : {s['junctions']}  (unexpected branch points: {s['branch_points']})")
    print(f"Sheet border   : {s['sheet_frames']} frame(s); excluded {s['border_lines_excluded']} line(s), "
          f"{s['border_texts_excluded']} margin text(s), {s['table_components_dropped']} table component(s)")
    for p in data["_written"]:
        print(f"Wrote          : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
