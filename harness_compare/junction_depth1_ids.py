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
from dataclasses import asdict, dataclass
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

@dataclass
class EngineConfig:
    # Wire extraction
    wire_width: Optional[float] = 0.539      # None -> auto-detect
    wire_width_tol: float = 0.03
    min_wire_length: float = 0.25
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


def extract_text_objects(page, M, cfg: EngineConfig) -> List[dict]:
    """Extract text objects (word groups) in the displayed frame."""
    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
    raw = page.get_text("rawdict", flags=flags)
    out: List[dict] = []
    tol = math.sin(math.radians(cfg.text_axis_tol_deg))
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            ld = line.get("dir", (1.0, 0.0))
            vd = unit(vec_to_view(M, ld))
            for span in line.get("spans", []):
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
    if w <= 0.1 or h <= 0.1:
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


def auto_wire_width(page, M, drawings: Optional[list] = None) -> Optional[float]:
    """Pick the stroke width carrying the largest total length of long black lines."""
    totals: Dict[float, float] = defaultdict(float)
    for d in (drawings if drawings is not None else page.get_drawings()):
        if "s" not in (d.get("type") or "") or _is_dashed(d.get("dashes")) or _is_white(d.get("color")):
            continue
        w = round(float(d.get("width") or 0.0), 3)
        if w < 0.2:
            continue
        for it in d.get("items", []):
            if it[0] == "l":
                L = dist(to_view(M, it[1]), to_view(M, it[2]))
                if L >= 8.0:
                    totals[w] += L
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def extract_vectors(page, M, cfg: EngineConfig) -> dict:
    """Extract wires, raw circles, other (non-wire) lines and closed symbol shapes."""
    drawings = page.get_drawings()
    wire_width = cfg.wire_width
    if wire_width is None:
        wire_width = auto_wire_width(page, M, drawings)
        if wire_width is None:
            raise RuntimeError("Could not auto-detect the harness wire stroke width; pass --wire-width.")
    prect = page.rect
    margin = 5.0

    def on_page(p) -> bool:
        return (prect.x0 - margin <= p[0] <= prect.x1 + margin) and (prect.y0 - margin <= p[1] <= prect.y1 + margin)

    wires: List[dict] = []
    circles: List[dict] = []
    other_lines: List[dict] = []
    shapes: List[dict] = []
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
        is_wire = stroked and not dashed and abs(width - wire_width) <= cfg.wire_width_tol
        closed = False
        if len(segs) >= 3 and dist(segs[0][0], segs[-1][1]) <= 0.3:
            closed = True
        if any(k == "poly" for k, _ in view_items):
            closed = True
        for (a, b, ii, kind) in segs:
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
        if not is_wire and closed and segs:
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
    return {"wire_width": wire_width, "wires": wires, "circles_raw": circles,
            "other_lines": other_lines, "shapes": shapes}


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


def symbol_directions(sym: dict, wires: List[dict], wire_index: GridIndex, cfg: EngineConfig) -> List[dict]:
    """Unique directions in which wires leave a circle symbol."""
    c = sym["center"]
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
    return cluster_directions(vectors, cfg.direction_merge_deg)


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
        if s["outer_diameter"] < cfg.element_min_diameter or s["has_filled_circle"]:
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
    classify_nodes(graph, vec, cfg)
    return graph


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
        if o["width"] < cfg.terminal_symbol_min_width and not o.get("filled"):
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

    Returns (metrics, "ok") for a valid candidate, or (None, reason).
    PDF / displayed-frame y grows downwards, so "below" means larger y and
    "right" means larger x.
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
        side = cy - wire_c                      # positive = below
        near_edge_gap = y0 - wire_c
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
        side = cx - wire_c                      # positive = right
        near_edge_gap = x0 - wire_c
        along = cy
    if side <= cfg.label_min_side:
        return None, "wrong_side"
    if side > cfg.label_max_side:
        return None, "too_far"
    if near_edge_gap < -cfg.label_overlap_tol:
        return None, "overlaps_wire"
    length = hi - lo
    mid = (lo + hi) / 2.0
    half = max(length / 2.0, 1.0)
    centre_err = abs(along - mid)
    projection = 0.5 if length <= 1e-9 else (along - lo) / length
    score = side / cfg.label_max_side + cfg.centre_weight * min(centre_err / half, 2.0)
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
            close = [c for c in cands[1:] if abs(c["along"] - best["along"]) < 3.0
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
    """Run text + vector extraction, symbols, graph and depth-1 ids for one page."""
    M = page_matrix(page)
    texts = extract_text_objects(page, M, cfg)
    vec = extract_vectors(page, M, cfg)
    symbols = build_symbols(vec, cfg)
    graph = build_wire_graph(vec, symbols, cfg)
    junctions = compute_depth1_ids(graph, texts, cfg)
    return {"matrix": M, "texts": texts, "vectors": vec, "symbols": symbols,
            "graph": graph, "junctions": junctions}


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
    for w in ext["vectors"]["wires"]:
        cv2.line(out, zp(w["p0"], zoom), zp(w["p1"], zoom), (255, 120, 0), max(2, int(zoom)), cv2.LINE_AA)
    colors = {"horizontal": (0, 160, 0), "vertical": (200, 0, 200), "angled": (0, 120, 255)}
    for t in ext["texts"]:
        b = t["bbox"]
        col = colors[t["orientation"]] if t["is_numeric"] else (150, 150, 150)
        cv2.rectangle(out, zp((b[0], b[1]), zoom), zp((b[2], b[3]), zoom), col, 2 if t["is_numeric"] else 1)
    for n in ext["graph"]["nodes"]:
        if n["kind"] in ("terminal", "open", "branch", "element"):
            col = {"terminal": (0, 140, 0), "open": (0, 0, 255), "branch": (255, 0, 255),
                   "element": (180, 0, 180)}[n["kind"]]
            cv2.circle(out, zp(n["point"], zoom), max(5, int(2.5 * zoom)), col, 2, cv2.LINE_AA)
    legend = ["wires: blue", "numeric text: green=horizontal, magenta=vertical, orange=angled",
              "non-numeric text: grey", "nodes: green=terminal, red=open, magenta=branch, purple=element"]
    for i, s in enumerate(legend):
        put_label(out, s, (10, 22 + 20 * i), (0, 0, 0), 0.55)
    return out


def draw_junction_diagnostics(img: np.ndarray, ext: dict, zoom: float) -> np.ndarray:
    out = fade(img)
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
                    cv2.rectangle(out, zp((b[0], b[1]), zoom), zp((b[2], b[3]), zoom), col, 2)
        jp = zp(j["point"], zoom)
        cv2.circle(out, jp, max(5, int(2 * zoom)), (0, 0, 230), -1, cv2.LINE_AA)
        put_label(out, j["structural_id"], (jp[0] + 6, jp[1] - 6), (0, 0, 160), 0.45)
    legend = ["red=junction dot, orange=two-direction dot, yellow=endpoint dot, cyan=hollow symbol,",
              "purple=large element, coloured paths = traced outgoing directions, boxes = chosen labels"]
    for i, s in enumerate(legend):
        put_label(out, s, (10, 22 + 20 * i), (0, 0, 0), 0.55)
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def add_engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--wire-width", default="0.539",
                    help="Harness wire stroke width in PDF points, or 'auto' (default 0.539).")
    ap.add_argument("--wire-width-tol", type=float, default=0.03)
    ap.add_argument("--label-max-side", type=float, default=10.5,
                    help="Maximum distance (pt) from wire to label centre on the required side.")
    ap.add_argument("--projection-margin", type=float, default=2.0,
                    help="How far (pt) a label centre may project beyond a straight piece end.")


def engine_config_from_args(args) -> EngineConfig:
    cfg = EngineConfig()
    ww = str(args.wire_width).strip().lower()
    try:
        cfg.wire_width = None if ww == "auto" else float(ww)
    except ValueError:
        raise ValueError(f"--wire-width must be a number or 'auto' (got {args.wire_width!r})")
    cfg.wire_width_tol = float(args.wire_width_tol)
    cfg.label_max_side = float(args.label_max_side)
    cfg.projection_margin = float(args.projection_margin)
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
            "coordinate_frame": "displayed_page",
            "source_pdf": source_pdf_info(pdf, page),
            "config": cfg.to_dict(),
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
            },
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
    print(f"Wire width     : {data['wire_width_used']}")
    print(f"Wire vectors   : {s['wire_vectors']}   texts: {s['texts']} (numeric {s['numeric_texts']})")
    print(f"Circle symbols : {s['circle_symbols']}  {s['symbol_kinds']}")
    print(f"Graph          : {s['graph_nodes']} nodes, {s['graph_edges']} edges  {s['node_kinds']}")
    print(f"Junctions      : {s['junctions']}  (unexpected branch points: {s['branch_points']})")
    for p in data["_written"]:
        print(f"Wrote          : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
