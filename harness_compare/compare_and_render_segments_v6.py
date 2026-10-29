#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 3 - compare_and_render_segments_v6.py

Strict text ownership and final structural comparison of two harness /
wiring-diagram PDFs.

This file is stand-alone: it only needs the two PDFs and the two segment JSON
files produced from THOSE PDFs (normally by build_logical_segments.py).

Example
-------
    python compare_and_render_segments_v6.py \\
        --pdf-a "old.pdf" --pdf-b "new.pdf" \\
        --segments-a "segments_pair/pdf_a_segments.json" \\
        --segments-b "segments_pair/pdf_b_segments.json" \\
        --out "comparison_output"

Processing outline (done separately for PDF A and PDF B, then compared)
-----------------------------------------------------------------------
 1. Validate that each segment JSON belongs to its PDF (SHA-256 fingerprint
    when present, otherwise a geometric check of the stored polylines against
    the PDF vectors; the coordinate frame is detected at the same time).
 2. Re-extract all text objects: value, normalized value, centre, bbox,
    direction, orientation, numeric flag, unique ID.
 3. Divide every segment polyline into straight wire pieces (also split at
    breakpoints / leader feet / ticks found on the wire) and classify each piece
    as horizontal, vertical or angled.  Bends stay polyline points only.
 4. Build every valid numeric-text-to-piece candidate on the whole page:
        horizontal piece: horizontal numeric text, BELOW the wire (larger y),
                          close to it, centre projecting onto the piece;
        vertical piece:   vertical (rotated) numeric text, RIGHT of the wire
                          (larger x), close to it, centre projecting onto it;
        angled piece:     never claims a label.
    Score = perpendicular distance + centre-alignment error (lower is better).
 5. Sort all candidates globally and give every text object to exactly one
    piece (global ownership).  A text is never reused by another piece.
 6. Rebuild each ordered segment sequence by cumulative distance along the
    whole polyline, keeping the original sequence as a fallback.
 7. Improve terminal / element identities with globally owned, meaningful
    nearby text (generic 2M / 8M, numbers, spec codes, annotations rejected)
    and rebuild the segment keys.  Junction depth-1 IDs are rebuilt from the
    corrected sequences while the original IDs are retained.
 8. Score every A/B segment pair (endpoint compatibility in both directions,
    junction multiset similarity with UNLABELLED wildcards, terminal identity,
    content similarity forward / reversed by edit distance, wire-number
    similarity) and match one-to-one (greedy, best first).
 9. Classify: same exact, same probable, changed, annotation / extraction
    review, entire branch removed, entire branch added.  Route merges / splits
    caused by a removed or added junction are recovered as review items before
    anything is called removed or added.
10. Write corrected segment JSON files, segment_comparison_v6.json and the
    two marked images.

All geometry is handled in the DISPLAYED page frame (page /Rotate applied,
origin top-left, y downwards).  Absolute positions are used only for
extraction, text association and drawing - never for matching.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # PyMuPDF >= 1.24 exposes "pymupdf"; older versions only "fitz".
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

# Text boxes from nominal ascender / descender (0.8 / -0.2 of the font size) rather than
# per-font corrections: the same box geometry for every font, and the same results whatever
# PyMuPDF add-ons are installed (the layout and table add-ons switch this global on import).
if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "unset_quad_corrections"):
    fitz.TOOLS.unset_quad_corrections(True)

VERSION = "v6"

# ===========================================================================
# Configuration
# ===========================================================================


# Values measured on the reference drawing the thresholds were tuned on.  Every distance
# below is the reference value; calibrate_pdf() measures each PDF and derive_pdf_config()
# rescales the distances to that PDF's own units.
REF_TEXT_UNIT = 5.39      # median font size of numeric labels (pt)
REF_WIRE_UNIT = 0.539     # harness wire stroke width (pt)
REF_DOT_UNIT = 2.16       # junction dot outer diameter (pt)
REF_LABEL_DISTANCE = 5.04 # median distance of a wire number's centre from its wire (pt)


@dataclass
class V6Config:
    # Calibration (None = measured per PDF)
    wire_width: Optional[float] = None      # explicit wire pen; else the pen of the lines under the segments
    wire_width_auto: bool = True
    wire_width_tol: float = 0.03
    label_max_side_override: Optional[float] = None
    side_h: Optional[int] = None            # +1 labels below horizontal wires, -1 above, 0 on the wire
    side_v: Optional[int] = None            # +1 labels right of vertical wires, -1 left, 0 on the wire
    deskew: bool = True
    text_unit: float = REF_TEXT_UNIT
    wire_unit: float = REF_WIRE_UNIT
    dot_unit: float = REF_DOT_UNIT
    layout_unit: float = REF_TEXT_UNIT
    inline_max_side: float = 3.2
    significant_min_width: float = 0.22     # non-wire lines thinner AND shorter are texture (hatching)
    significant_min_length: float = 3.2
    end_margin: float = 1.0                 # stations closer than this to a route end are ignored
    group_tol: float = 1.5                  # segment ends this close share one node
    missing_position_window: float = 25.0
    # Orientation tolerances
    axis_tol_deg: float = 8.0           # straight wire piece horizontal / vertical
    text_axis_tol_deg: float = 12.0     # text horizontal / vertical
    text_gap_factor: float = 0.8
    # Strict numeric label placement
    label_min_side: float = 0.4         # label centre at least this far on the required side
    label_max_side: float = 10.5        # ... and at most this far (small configured maximum)
    label_overlap_tol: float = 1.9      # text box may not cross the wire by more than this
    projection_margin: float = 2.0      # centre may project this far beyond a piece end
    weight_perpendicular: float = 1.0
    weight_centre: float = 0.5
    label_slot_gap: float = 0.5         # two labels on one piece must not overlap along it
    # Stations (points that divide label spans on a straight run)
    station_tol: float = 0.8
    station_line_max_length: float = 40.0
    station_merge_dist: float = 2.0
    small_circle_max_diameter: float = 3.0
    # Terminal / element identity
    terminal_text_radius: float = 30.0
    element_text_radius: float = 22.0
    terminal_below_penalty: float = 6.0
    terminal_below_slack: float = 3.0
    terminal_inward_penalty: float = 8.0
    terminal_align_weight: float = 0.25     # per pt off the terminal's line (reading frame)
    terminal_min_confidence: float = 0.35
    # Extraction-uncertainty radius (a value visible near the other segment)
    uncertain_text_radius: float = 15.0
    # JSON-to-PDF ownership check
    geometry_match_tol: float = 1.5
    min_geometry_match: float = 0.6
    # Segment matching
    weight_endpoint: float = 0.6
    weight_content: float = 0.25
    weight_wire: float = 0.15
    min_pair_score: float = 0.45
    confident_pair_score: float = 0.7
    junction_min_common: int = 2
    node_anchor_min: float = 0.6          # node pairs this similar anchor neighbour support
    node_corr_min: float = 0.6            # minimum score of an accepted node correspondence
    strong_endpoint_min: float = 0.6
    strong_endpoint_mean: float = 0.75
    partial_endpoint_max: float = 0.8
    partial_content_min: float = 0.6
    content_driven_min: float = 0.85
    topology_content_min: float = 0.75
    topology_endpoint_min: float = 0.55
    terminal_change_min_confidence: float = 0.6
    component_orientation_min_strength: float = 0.15   # min handedness to call a symbol asymmetric
    # Sheet border (drawing frame, zone markers)
    border_detection: bool = True
    border_min_fraction: float = 0.6
    border_corner_tol: float = 2.5
    border_segment_fraction: float = 0.9   # JSON segments this much on the border are ignored
    # Output
    mark_same: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Text classification
# ===========================================================================

# Wire number: digits with an optional decimal component (52, 52.0, 35.8)
NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")
# Generic terminal-size / route codes: 1M, 2M, 8M, 2M,8M
GENERIC_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?M(?:\s*[,/]\s*\d+(?:\.\d+)?M)*$", re.IGNORECASE)
# Wire / part specification codes: EN6049-007, NSA935805-06, ABS2418A05-5, D924-95612-051
SPEC_CODE_RES = (
    re.compile(r"^[A-Z]{2,5}\d{3,}[A-Z0-9]*(?:-[A-Z0-9]+)+$"),
    re.compile(r"^[A-Z]\d{3,}(?:-\d{2,})+$"),
)
ANNOTATION_WORDS = ("CLAMP", "MARKER", "ROUTE", "LABEL", "BRAID", "SUPPORT", "SPLIT SUPPORT",
                    "SLEEVE", "TIE", "TAPE", "GROMMET", "SHIELD")
UNLABELLED = "UNLABELLED"
STRUCT_ID_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]\s*$")


def normalize_text(raw) -> str:
    s = " ".join(str(raw).split())
    if NUMERIC_RE.match(s) and "." in s:
        whole, frac = s.split(".", 1)
        if set(frac) <= {"0"}:
            s = whole
    return s


def is_numeric_text(s: str) -> bool:
    return bool(NUMERIC_RE.match(s))


def is_generic_size(s: str) -> bool:
    return bool(GENERIC_SIZE_RE.match(s.replace(" ", "")))


def is_spec_code(s: str) -> bool:
    t = s.replace(" ", "").upper()
    return any(r.match(t) for r in SPEC_CODE_RES)


def annotation_keyword(s: str) -> Optional[str]:
    up = str(s).upper().replace("_", " ")
    if "SPLIT SUPPORT" in up:
        return "SUPPORT"
    for w in ANNOTATION_WORDS:
        if re.search(r"(?<![A-Z])" + re.escape(w) + r"(?![A-Z])", up):
            return "SUPPORT" if w == "SPLIT SUPPORT" else w
    return None


def is_meaningful_identifier(s: str) -> bool:
    """Terminal / element names (7554VC-A, 6711VN/B2, 14MP-A, B1) - not 2M, 52, EN6049-007, CLAMP."""
    t = s.strip()
    if len(t) < 2 or is_numeric_text(t) or is_generic_size(t) or is_spec_code(t) or annotation_keyword(t):
        return False
    return bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"\d", t))


def canonical_name(name: str) -> str:
    return re.sub(r"\s*/\s*", "/", " ".join(str(name).split())).upper()


def numeric_sort_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(p, a, b) -> Tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return dist(p, a), 0.0
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)), t


def unit(v) -> Tuple[float, float]:
    L = math.hypot(v[0], v[1])
    return (0.0, 0.0) if L <= 1e-12 else (v[0] / L, v[1] / L)


def rect_distance(p, r) -> float:
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def piece_orientation(p0, p1, tol_deg: float) -> str:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) <= 1e-9:
        return "point"
    a = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    if a <= tol_deg or a >= 180.0 - tol_deg:
        return "horizontal"
    if abs(a - 90.0) <= tol_deg:
        return "vertical"
    return "angled"


def polyline_length(poly) -> float:
    return sum(dist(a, b) for a, b in zip(poly, poly[1:]))


def project_on_polyline(p, poly) -> Tuple[float, float, Tuple[float, float]]:
    """(distance, arc-length position, foot point)."""
    best = (float("inf"), 0.0, (poly[0][0], poly[0][1]))
    cum = 0.0
    for a, b in zip(poly, poly[1:]):
        d, t = point_segment_distance(p, a, b)
        L = dist(a, b)
        if d < best[0]:
            best = (d, cum + t * L, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        cum += L
    return best


def touch_or_cross(a, b, p, q, tol) -> Optional[Tuple[float, float]]:
    """Point where segment ab touches (endpoint within tol) or crosses segment pq."""
    for e in (a, b):
        d, t = point_segment_distance(e, p, q)
        if d <= tol:
            return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
    d1x, d1y = b[0] - a[0], b[1] - a[1]
    d2x, d2y = q[0] - p[0], q[1] - p[1]
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-9:
        return None
    t = ((p[0] - a[0]) * d2y - (p[1] - a[1]) * d2x) / den
    u = ((p[0] - a[0]) * d1y - (p[1] - a[1]) * d1x) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (a[0] + t * d1x, a[1] + t * d1y)
    return None


def rnd(v: float, nd: int = 3) -> float:
    return float(round(float(v), nd))


def rpt(p, nd: int = 3) -> List[float]:
    return [rnd(p[0], nd), rnd(p[1], nd)]


class GridIndex:
    """Uniform-grid spatial index for bounding boxes."""

    def __init__(self, cell: float = 25.0):
        self.cell = float(cell)
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def _cells(self, x0, y0, x1, y1):
        c = self.cell
        for gx in range(int(math.floor(min(x0, x1) / c)), int(math.floor(max(x0, x1) / c)) + 1):
            for gy in range(int(math.floor(min(y0, y1) / c)), int(math.floor(max(y0, y1) / c)) + 1):
                yield (gx, gy)

    def insert(self, key: int, x0, y0, x1, y1) -> None:
        for cell in self._cells(x0, y0, x1, y1):
            self.buckets[cell].append(key)

    def query(self, x0, y0, x1, y1) -> List[int]:
        seen, out = set(), []
        for cell in self._cells(x0, y0, x1, y1):
            for k in self.buckets.get(cell, ()):
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out


# ===========================================================================
# PDF extraction (text, line vectors, small circles)
# ===========================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_view(M, p) -> Tuple[float, float]:
    """Affine transform with plain floats (much faster than fitz.Point * Matrix)."""
    x, y = float(p[0]), float(p[1])
    return (M.a * x + M.c * y + M.e, M.b * x + M.d * y + M.f)


def extract_texts(page, cfg: V6Config, M=None, view=None) -> List[dict]:
    """All text objects (word groups) in the analysis frame, each with a unique ID."""
    M = page.rotation_matrix if M is None else M
    # read without clipping and keep whole words that are at least partly visible, so a name
    # cut by the crop box is not shortened, and other sheets placed off the page are ignored
    raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
                        clip=fitz.INFINITE_RECT())
    visible = page.rect if view is None else view
    tol = math.sin(math.radians(cfg.text_axis_tol_deg))
    out = []
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            ld = line.get("dir", (1.0, 0.0))
            vd = unit((M.a * ld[0] + M.c * ld[1], M.b * ld[0] + M.d * ld[1]))
            for span in line.get("spans", []):
                if not (fitz.Rect(span["bbox"]) * M).intersects(visible):
                    continue                  # text of another sheet placed off the page
                size = float(span.get("size", 0.0))
                groups: List[List[dict]] = []
                cur: List[dict] = []
                prev = None
                for ch in span.get("chars") or []:
                    if not ch.get("c", "").strip():
                        if cur:
                            cur.append(ch)
                        continue
                    r = fitz.Rect(ch["bbox"])
                    if prev is not None:
                        pr = fitz.Rect(prev["bbox"])
                        along = (((r.x0 + r.x1) - (pr.x0 + pr.x1)) * ld[0] + ((r.y0 + r.y1) - (pr.y0 + pr.y1)) * ld[1]) / 2
                        ext = (abs(ld[0]) * (pr.width + r.width) + abs(ld[1]) * (pr.height + r.height)) / 2
                        if along - ext > cfg.text_gap_factor * max(size, 1.0) or along < -0.5 * max(size, 1.0):
                            groups.append(cur)
                            cur = []
                    cur.append(ch)
                    prev = ch
                if cur:
                    groups.append(cur)
                for g in groups:
                    raw_text = "".join(c.get("c", "") for c in g).strip()
                    norm = normalize_text(raw_text)
                    if not norm:
                        continue
                    rect = fitz.Rect()
                    for c in g:
                        if c.get("c", "").strip():
                            rect |= fitz.Rect(c["bbox"]) * M
                    if not rect.intersects(visible):
                        continue
                    if abs(vd[1]) <= tol:
                        orientation = "horizontal"
                    elif abs(vd[0]) <= tol:
                        orientation = "vertical"
                    else:
                        orientation = "angled"
                    out.append({
                        "id": "", "text": raw_text, "norm": norm,
                        "center": [rnd((rect.x0 + rect.x1) / 2), rnd((rect.y0 + rect.y1) / 2)],
                        "bbox": [rnd(rect.x0), rnd(rect.y0), rnd(rect.x1), rnd(rect.y1)],
                        "dir": [rnd(vd[0], 4), rnd(vd[1], 4)],
                        "orientation": orientation,
                        "is_numeric": is_numeric_text(norm),
                        "font_size": rnd(size, 2),
                    })
    out.sort(key=lambda t: (t["center"][1], t["center"][0], t["norm"]))
    for i, t in enumerate(out):
        t["id"] = f"T{i:05d}"
    return out


def _is_dashed(dashes) -> bool:
    s = str(dashes or "").strip()
    return bool(re.search(r"\[\s*[\d.]", s))


def _is_white(color) -> bool:
    try:
        return color is not None and all(float(c) >= 0.97 for c in color)
    except TypeError:
        return False


def _bezier(p0, p1, p2, p3, n=6):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        pts.append((mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                    mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]))
    return pts


def _merge_collinear(lines, ctol: float = 0.6, gap: float = 2.0):
    out = []
    for lo, hi, c in sorted(lines, key=lambda l: (l[2], l[0])):
        for m in out:
            if abs(m[2] - c) <= ctol and lo <= m[1] + gap and hi >= m[0] - gap:
                m[0], m[1] = min(m[0], lo), max(m[1], hi)
                break
        else:
            out.append([lo, hi, c])
    out.sort(key=lambda m: -(m[1] - m[0]))
    return [tuple(m) for m in out]


def detect_sheet_border(page, drawings: list, text_centres: List[Tuple[float, float]], cfg: V6Config,
                        M=None) -> dict:
    """Drawing frame(s): large axis-aligned rectangles framing the sheet (any stroke width).

    Same rule as Script 1: one rectangle path or four lines, >= border_min_fraction of the
    page in both directions, at most about one page in size, overlapping at least half the
    page and enclosing most of the page text.  The innermost frame bounds the drawing area.
    """
    W, H = page.rect.width, page.rect.height
    if not cfg.border_detection:
        return {"frames": [], "inner": None}
    M = page.rotation_matrix if M is None else M
    U = cfg.layout_unit
    ax_tol = max(0.5 * U / REF_TEXT_UNIT, 0.05)
    rects, hs, vs = [], [], []
    for d in drawings:
        if "s" not in (d.get("type") or "") or _is_white(d.get("color")):
            continue
        for it in d.get("items", []):
            if it[0] == "l":
                segs = [(to_view(M, it[1]), to_view(M, it[2]))]
            elif it[0] in ("re", "qu"):
                q = fitz.Rect(it[1]).quad if it[0] == "re" else it[1]
                pts = [to_view(M, v) for v in (q.ul, q.ur, q.lr, q.ll, q.ul)]
                segs = list(zip(pts, pts[1:]))
                if all(abs(a[0] - b[0]) <= ax_tol or abs(a[1] - b[1]) <= ax_tol for a, b in segs):
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    rects.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                continue
            for a, b in segs:
                if abs(a[1] - b[1]) <= ax_tol and abs(a[0] - b[0]) >= 0.5 * W:
                    hs.append((min(a[0], b[0]), max(a[0], b[0]), (a[1] + b[1]) / 2))
                elif abs(a[0] - b[0]) <= ax_tol and abs(a[1] - b[1]) >= 0.5 * H:
                    vs.append((min(a[1], b[1]), max(a[1], b[1]), (a[0] + b[0]) / 2))
    hs, vs = _merge_collinear(hs, 0.11 * U, 0.37 * U)[:16], _merge_collinear(vs, 0.11 * U, 0.37 * U)[:16]
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
    frames = []
    for r in rects:
        w, h = r[2] - r[0], r[3] - r[1]
        if w < cfg.border_min_fraction * W or h < cfg.border_min_fraction * H or w > 1.15 * W or h > 1.15 * H:
            continue
        ix = max(0.0, min(r[2], W) - max(r[0], 0.0))
        iy = max(0.0, min(r[3], H) - max(r[1], 0.0))
        if ix * iy < 0.5 * W * H:
            continue
        if text_centres and sum(1 for c in text_centres if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]) \
                < 0.75 * len(text_centres):
            continue
        if any(all(abs(r[k] - f[k]) <= 0.19 * U for k in range(4)) for f in frames):
            continue
        frames.append(r)
    frames.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    return {"frames": [tuple(rnd(v) for v in f) for f in frames], "inner": frames[-1] if frames else None,
            "band_tol": max(1.86 * cfg.wire_unit, 0.19 * U), "text_tol": 0.093 * U}


def _outside(p, r, tol: float) -> bool:
    return p[0] < r[0] - tol or p[0] > r[2] + tol or p[1] < r[1] - tol or p[1] > r[3] + tol


def _inside(p, r, tol: float) -> bool:
    return r[0] + tol < p[0] < r[2] - tol and r[1] + tol < p[1] < r[3] - tol


def is_border_segment(a, b, border: Optional[dict], tol: Optional[float] = None) -> bool:
    """A frame edge, or a line in the margin band outside the drawing area (zone ticks)."""
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


def extract_vectors(page, cfg: V6Config, drawings: Optional[list] = None, border: Optional[dict] = None,
                    M=None) -> dict:
    """Stroked line pieces (all widths) and small circles in the analysis frame.

    Sheet-border lines are kept for the ownership geometry check but flagged, so they are
    never used as stations.  Wire / significance flags are set later by flag_lines() once
    the wire pen of this PDF is known.
    """
    M = page.rotation_matrix if M is None else M
    drawings = page.get_drawings() if drawings is None else drawings
    lines, unrot_pts, small_circles = [], [], []
    for di, d in enumerate(drawings):
        dtype = d.get("type") or ""
        width = float(d.get("width") or 0.0)
        stroked = "s" in dtype and not _is_white(d.get("color"))
        filled = "f" in dtype and d.get("fill") is not None and not _is_white(d.get("fill"))
        if not stroked and not filled:
            continue
        dashed = _is_dashed(d.get("dashes"))
        items = d.get("items", [])
        curves = [it for it in items if it[0] == "c"]
        if len(curves) >= 2 and all(it[0] in ("c", "l") for it in items):
            samples = []
            for it in curves:
                samples += _bezier(*[to_view(M, q) for q in it[1:5]])
            xs, ys = [p[0] for p in samples], [p[1] for p in samples]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if 0.02 * cfg.text_unit < w <= 3.0 * cfg.text_unit and h > 0.02 * cfg.text_unit and abs(w / h - 1) < 0.2:
                small_circles.append({"center": ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2),
                                      "diameter": (w + h) / 2, "filled": filled})
            continue
        if not stroked:
            continue
        for it in items:
            segs = []
            if it[0] == "l":
                segs.append((it[1], it[2]))
            elif it[0] == "re":
                r = fitz.Rect(it[1])
                segs += [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
            elif it[0] == "qu":
                q = it[1]
                segs += [(q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)]
            for a, b in segs:
                va, vb = to_view(M, a), to_view(M, b)
                if dist(va, vb) < 0.05:
                    continue
                lines.append({"p0": va, "p1": vb, "width": width, "dashed": dashed, "is_wire": False,
                              "significant": False, "border": is_border_segment(va, vb, border)})
                unrot_pts.append(((float(a.x), float(a.y)), (float(b.x), float(b.y))))
    return {"lines": lines, "unrotated_lines": unrot_pts, "small_circles": small_circles}


def flag_lines(vec: dict, cfg: V6Config) -> None:
    """Wire pen lines, and significant non-wire lines (symbols, leaders, ticks) vs texture."""
    for l in vec["lines"]:
        l["is_wire"] = (cfg.wire_width is not None and not l["dashed"]
                        and abs(l["width"] - cfg.wire_width) <= max(cfg.wire_width_tol, 0.06 * cfg.wire_width))
        l["significant"] = (not l["dashed"] and not l["is_wire"] and not l["border"]
                            and (l["width"] >= cfg.significant_min_width
                                 or dist(l["p0"], l["p1"]) >= cfg.significant_min_length))


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


# ===========================================================================
# Per-PDF calibration (orientation, units, label convention)
# ===========================================================================

def _solid_stroke(d: dict) -> bool:
    return "s" in (d.get("type") or "") and not _is_dashed(d.get("dashes")) and not _is_white(d.get("color"))


def dominant_axis_angle(drawings: list, M, page) -> Tuple[float, float]:
    """Angle (deg) by which the drawing is rotated off the page axes, and its support (same
    rule as Script 1: length-weighted histogram of long solid lines folded modulo 90 deg)."""
    min_len = 0.01 * max(page.rect.width, page.rect.height)
    hist = [0.0] * 180
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


def make_frame(page, cfg: V6Config, drawings: list) -> dict:
    """Analysis frame = displayed page de-skewed about its centre (identity for normal drawings)."""
    M0 = page.rotation_matrix
    theta, support = dominant_axis_angle(drawings, M0, page) if cfg.deskew else (0.0, 0.0)
    if abs(theta) < 0.3 or support < 0.5:
        theta = 0.0
    if theta == 0.0:
        ident = fitz.Matrix(1, 0, 0, 1, 0, 0)
        return {"theta": 0.0, "support": support, "M": M0, "R": ident, "Rinv": ident, "view": fitz.Rect(page.rect)}
    cx, cy = page.rect.width / 2.0, page.rect.height / 2.0
    c, s_ = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    R = fitz.Matrix(1, 0, 0, 1, -cx, -cy) * fitz.Matrix(c, -s_, s_, c, 0, 0) * fitz.Matrix(1, 0, 0, 1, cx, cy)
    view = fitz.Rect()
    for q in (page.rect.tl, page.rect.tr, page.rect.br, page.rect.bl):
        view |= q * R
    return {"theta": theta, "support": support, "M": M0 * R, "R": R, "Rinv": ~R, "view": view}


def apply_matrix(T, p) -> Tuple[float, float]:
    return (T.a * p[0] + T.c * p[1] + T.e, T.b * p[0] + T.d * p[1] + T.f)


def wire_pen_from_segments(segs: List[dict], lines: List[dict], tol: float) -> Optional[float]:
    """Stroke width of the PDF lines lying along the segment polylines (length weighted)."""
    idx = GridIndex(25.0)
    for i, l in enumerate(lines):
        idx.insert(i, l["p0"][0], l["p0"][1], l["p1"][0], l["p1"][1])
    by_width: Dict[float, float] = defaultdict(float)
    for s in segs:
        for a, b in zip(s["polyline"], s["polyline"][1:]):
            if dist(a, b) < 4 * tol:
                continue
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            u = unit((b[0] - a[0], b[1] - a[1]))
            for li in idx.query(m[0] - tol, m[1] - tol, m[0] + tol, m[1] + tol):
                l = lines[li]
                if l["dashed"] or l["width"] <= 0:
                    continue
                v = unit((l["p1"][0] - l["p0"][0], l["p1"][1] - l["p0"][1]))
                if abs(u[0] * v[0] + u[1] * v[1]) < 0.999:
                    continue
                if point_segment_distance(m, l["p0"], l["p1"])[0] <= tol:
                    by_width[round(l["width"], 3)] += dist(l["p0"], l["p1"])
    return max(by_width.items(), key=lambda kv: kv[1])[0] if by_width else None


def label_text_unit(segs: List[dict], texts: List[dict]) -> Optional[float]:
    """Median font size of the numeric texts sitting parallel next to the segment routes (the
    wire numbers).  Zone numbers, table entries and notes in other fonts do not count."""
    pcs = [(a, b) for s in segs for a, b in zip(s["polyline"], s["polyline"][1:]) if dist(a, b) > 0]
    idx = GridIndex(30.0)
    for i, (a, b) in enumerate(pcs):
        idx.insert(i, a[0], a[1], b[0], b[1])
    fonts = []
    for t in texts:
        f = t.get("font_size", 0)
        if not t["is_numeric"] or f <= 0 or t["orientation"] not in ("horizontal", "vertical"):
            continue
        cx, cy = t["center"]
        for i in idx.query(cx - 3 * f, cy - 3 * f, cx + 3 * f, cy + 3 * f):
            a, b = pcs[i]
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if t["orientation"] == "horizontal" and dy <= 0.1 * dx:
                lo, hi, along, perp = min(a[0], b[0]), max(a[0], b[0]), cx, abs(cy - (a[1] + b[1]) / 2)
            elif t["orientation"] == "vertical" and dx <= 0.1 * dy:
                lo, hi, along, perp = min(a[1], b[1]), max(a[1], b[1]), cy, abs(cx - (a[0] + b[0]) / 2)
            else:
                continue
            if lo - 0.4 * f <= along <= hi + 0.4 * f and perp <= 3 * f:
                fonts.append(f)
                break
    fonts.sort()
    return fonts[len(fonts) // 2] if len(fonts) >= 5 else None


def calibrate_labels(segs: List[dict], texts: List[dict], text_unit: float, axis_tol_deg: float) -> dict:
    """Label side per wire orientation and typical distance, measured against the segment routes
    (same rule as Script 1: majority side of the nearest parallel piece)."""
    U = text_unit
    tn = math.tan(math.radians(axis_tol_deg))
    pcs = [(a, b) for s in segs for a, b in zip(s["polyline"], s["polyline"][1:]) if dist(a, b) > 0]
    idx = GridIndex(30.0)
    for i, (a, b) in enumerate(pcs):
        idx.insert(i, a[0], a[1], b[0], b[1])
    raws: Dict[str, List[float]] = {"horizontal": [], "vertical": []}
    for t in texts:
        if not t["is_numeric"] or t["orientation"] not in raws:
            continue
        cx, cy = t["center"]
        best = None
        for i in idx.query(cx - 4 * U, cy - 4 * U, cx + 4 * U, cy + 4 * U):
            a, b = pcs[i]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if t["orientation"] == "horizontal" and abs(dy) <= tn * abs(dx):
                lo, hi, along = min(a[0], b[0]), max(a[0], b[0]), cx
                raw = cy - (a[1] + (cx - a[0]) / dx * dy if abs(dx) > 1e-9 else a[1])
            elif t["orientation"] == "vertical" and abs(dx) <= tn * abs(dy):
                lo, hi, along = min(a[1], b[1]), max(a[1], b[1]), cy
                raw = cx - (a[0] + (cy - a[1]) / dy * dx if abs(dy) > 1e-9 else a[0])
            else:
                continue
            if lo - 0.4 * U <= along <= hi + 0.4 * U and abs(raw) <= 4 * U and (best is None or abs(raw) < abs(best)):
                best = raw
        if best is not None:
            raws[t["orientation"]].append(best)
    eps = 0.15 * U
    conv, evidence = {}, {}
    for o, rs in raws.items():
        n = len(rs)
        pos = sum(1 for r in rs if r > eps)
        neg = sum(1 for r in rs if r < -eps)
        evidence[o] = {"samples": n, "positive": pos, "negative": neg, "on_wire": n - pos - neg}
        conv[o] = (None if n < 4 else 1 if pos >= 0.65 * n else -1 if neg >= 0.65 * n
                   else 0 if (n - pos - neg) >= 0.5 * n else None)
    dists = []
    for o, rs in raws.items():
        c = conv[o] if conv[o] is not None else 1
        dists += [abs(r) for r in rs if (c == 0 and abs(r) <= 3 * eps) or (c != 0 and c * r > eps)]
    dists.sort()
    return {"side_h": conv["horizontal"], "side_v": conv["vertical"],
            "median_distance": dists[len(dists) // 2] if dists else None, "evidence": evidence}


def derive_pdf_config(base: V6Config, prof: dict) -> V6Config:
    """Every distance threshold rescaled from the reference drawing to this PDF's units."""
    Ut, Uw, D, UL = prof["text_unit"], prof["wire_unit"], prof["dot_unit"], prof["layout_unit"]
    g = UL / REF_TEXT_UNIT
    c = replace(base)
    c.text_unit, c.wire_unit, c.dot_unit, c.layout_unit = Ut, Uw, D, UL
    c.wire_width = prof["wire_width"]
    c.wire_width_tol = max(0.02, 0.06 * Uw)
    c.text_axis_tol_deg = prof["text_axis_tol_deg"]
    c.label_min_side = 0.075 * Ut
    c.label_max_side = prof["label_max_side"]
    c.label_overlap_tol = 0.35 * Ut
    c.projection_margin = 0.37 * Ut
    c.inline_max_side = prof["inline_max_side"]
    c.side_h, c.side_v = prof["side_h"], prof["side_v"]
    c.label_slot_gap = 0.093 * Ut
    c.station_tol = max(1.5 * Uw, 0.15 * UL)
    c.station_line_max_length = 40.0 * g
    c.station_merge_dist = 2.0 * g
    c.small_circle_max_diameter = 1.39 * D
    # between hatching (2.5 % of the layout unit) and leaders (6 %) on the reference drawing, with
    # the same margin both ways; independent of the wire pen, so a heavy or fine wire pen does
    # not turn leaders into texture or hatching into leaders
    c.significant_min_width = 0.04 * UL
    c.significant_min_length = 0.6 * UL
    c.end_margin = 1.0 * g
    c.terminal_text_radius = 30.0 * g
    c.element_text_radius = 22.0 * g
    c.terminal_below_penalty = 6.0 * g
    c.terminal_below_slack = 3.0 * g
    c.terminal_inward_penalty = 8.0 * g
    c.uncertain_text_radius = 15.0 * g
    c.geometry_match_tol = max(2.8 * Uw, 0.28 * UL)
    c.group_tol = max(2.8 * Uw, 0.28 * UL)
    c.missing_position_window = 25.0 * g
    c.border_corner_tol = 0.46 * UL
    return c


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
    devs = []
    for t in texts:
        a = math.degrees(math.atan2(t["dir"][1], t["dir"][0]))
        devs.append(min(abs(((a + 45.0) % 90.0) - 45.0), 45.0))
    devs.sort()
    if not devs:
        return 12.0
    return max(12.0, min(25.0, devs[min(len(devs) - 1, int(0.99 * len(devs)))] + 3.0))


# ===========================================================================
# Segment JSON loading, normalization and ownership validation
# ===========================================================================

class OwnershipError(RuntimeError):
    pass


TYPE_ALIASES = {
    "junction": "junction", "dot": "junction", "junction_dot": "junction", "branch": "junction",
    "branch_point": "junction", "terminal": "terminal", "connector": "terminal", "endpoint": "terminal",
    "element": "element", "circle": "element", "component": "element", "circular_element": "element",
    "open": "open", "open_endpoint": "open", "loop": "other",
}
PREFIX_TYPES = {
    "DOT1": "junction", "DOT": "junction", "JUNCTION": "junction", "BRANCH1": "junction", "BRANCH": "junction",
    "TERMINAL": "terminal", "TERM": "terminal", "CONNECTOR": "terminal",
    "ELEMENT": "element", "CIRCLE": "element", "COMPONENT": "element",
    "OPEN": "open", "END": "open",
}


def parse_structural_id(sid: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not sid:
        return None, None
    m = STRUCT_ID_RE.match(str(sid))
    if not m:
        return None, str(sid)
    return m.group(1).upper(), m.group(2)


def parse_junction_labels(inner: str) -> List[str]:
    out = []
    for part in str(inner).split("|"):
        p = normalize_text(part)
        if not p or p.upper() in ("UNLABELLED", "UNLABELED", "?", "NONE", "NULL"):
            out.append(UNLABELLED)
        else:
            out.append(p)
    return out


def junction_id(prefix: str, labels: List[str]) -> str:
    vals = sorted([l for l in labels if l != UNLABELLED], key=numeric_sort_key)
    return f"{prefix}[{'|'.join(vals + [UNLABELLED] * sum(1 for l in labels if l == UNLABELLED))}]"


def normalize_token(tok) -> Optional[str]:
    if isinstance(tok, dict):
        kind = str(tok.get("kind") or tok.get("type") or "").upper()
        value = tok.get("value", tok.get("text"))
    else:
        s = str(tok).strip()
        if not s:
            return None
        if ":" in s:
            kind, value = s.split(":", 1)
        else:
            kind, value = "", s
        kind = kind.upper().strip()
    value = normalize_text(value if value is not None else "")
    if not value:
        return None
    if kind in ("WIRE", "NUM", "NUMBER", "WIRE_NUMBER", "") and is_numeric_text(value):
        return f"WIRE:{value}"
    if kind == "WIRE" and not is_numeric_text(value):
        return f"ELEMENT:{value}"
    return f"ELEMENT:{value}"


def token_value(tok: str) -> str:
    return tok.split(":", 1)[1] if ":" in tok else tok


def token_kind(tok: str) -> str:
    return tok.split(":", 1)[0] if ":" in tok else "ELEMENT"


def is_annotation_token(tok: str) -> bool:
    return token_kind(tok) == "ELEMENT" and annotation_keyword(token_value(tok)) is not None


def _get_point(v) -> Optional[Tuple[float, float]]:
    if isinstance(v, dict):
        v = v.get("point") or v.get("xy")
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return None
    return None


def normalize_endpoint(raw, fallback_point) -> dict:
    if isinstance(raw, str):
        raw = {"structural_id": raw}
    raw = raw or {}
    sid = raw.get("structural_id") or raw.get("id") or raw.get("key") or raw.get("node_key")
    prefix, inner = parse_structural_id(sid)
    rtype = str(raw.get("type") or raw.get("kind") or "").lower()
    group = PREFIX_TYPES.get(prefix or "", None) or TYPE_ALIASES.get(rtype, "other")
    point = _get_point(raw.get("point")) or fallback_point
    node_point = _get_point(raw.get("node_point")) or point
    ep = {
        "type": group, "raw_type": rtype or None, "structural_id": sid,
        "original_structural_id": sid, "prefix": prefix, "point": point, "node_point": node_point,
        "node_id": raw.get("node_id"), "labels": None, "name": None,
        "name_confidence": raw.get("name_confidence"),
    }
    if group == "junction":
        if inner is not None:
            ep["labels"] = parse_junction_labels(inner)
        elif raw.get("depth1_labels"):
            ep["labels"] = [normalize_text(x) or UNLABELLED for x in raw["depth1_labels"]]
    elif group in ("terminal", "element"):
        name = raw.get("name") or inner
        if name and str(name).upper() not in ("UNIDENTIFIED", "UNNAMED", "?"):
            ep["name"] = canonical_name(name)
    return ep


def simplify_polyline(pts: List[Tuple[float, float]], collinear_deg: float = 0.5) -> List[Tuple[float, float]]:
    """Drop repeated points and vertices where the route does not bend (vector segmentation
    artefacts); real bends and the point order are kept."""
    clean: List[Tuple[float, float]] = []
    for p in pts:
        if not clean or dist(clean[-1], p) > 1e-3:
            clean.append(p)
    if len(clean) <= 2:
        return clean
    out = [clean[0]]
    cos_lim = math.cos(math.radians(collinear_deg))
    for i in range(1, len(clean) - 1):
        a, b, c = out[-1], clean[i], clean[i + 1]
        u, v = unit((b[0] - a[0], b[1] - a[1])), unit((c[0] - b[0], c[1] - b[1]))
        if u[0] * v[0] + u[1] * v[1] >= cos_lim:
            continue
        out.append(b)
    out.append(clean[-1])
    return out


def normalize_segment(raw: dict, i: int) -> dict:
    poly_raw = raw.get("polyline") or raw.get("points") or raw.get("path") or raw.get("geometry") or []
    poly = simplify_polyline([p for p in (_get_point(q) for q in poly_raw) if p is not None])
    if len(poly) < 2:
        raise ValueError(f"segment #{i} has no usable polyline")
    start = normalize_endpoint(raw.get("start"), poly[0])
    end = normalize_endpoint(raw.get("end"), poly[-1])
    items = []
    raw_items = raw.get("items")
    if isinstance(raw_items, list) and raw_items:
        for it in raw_items:
            tok = normalize_token(it)
            if tok is None:
                continue
            s = it.get("s") if isinstance(it, dict) else None
            ori = it.get("orientation") if isinstance(it, dict) else None
            items.append({"token": tok, "s": float(s) if isinstance(s, (int, float)) else None,
                          "text_id": it.get("text_id") if isinstance(it, dict) else None,
                          "annotation": is_annotation_token(tok),
                          "orientation": float(ori) if isinstance(ori, (int, float)) else None,
                          "caption": it.get("caption") if isinstance(it, dict) else None})
    else:
        for tok in raw.get("sequence") or raw.get("contents") or []:
            t = normalize_token(tok)
            if t is not None:
                items.append({"token": t, "s": None, "text_id": None, "annotation": is_annotation_token(t),
                              "caption": None})
    seq = raw.get("sequence")
    if isinstance(seq, list) and seq:
        orig_sequence = [t for t in (normalize_token(x) for x in seq) if t]
    else:
        orig_sequence = [it["token"] for it in items]
    return {
        "index": int(raw.get("segment_index", raw.get("index", i))),
        "raw": raw,
        "polyline": poly,
        "start": start,
        "end": end,
        "orig_items": items,
        "orig_sequence": orig_sequence,
        "orig_wires": [token_value(t) for t in orig_sequence if token_kind(t) == "WIRE"],
        "orig_key": raw.get("segment_key"),
        "vector_line_ids": list(raw.get("vector_line_ids") or raw.get("line_ids") or []),
        "json_stations": [p for p in (_get_point(st) for st in (raw.get("stations") or [])) if p],
    }


def load_segments_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Segment JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        data = {"segments": data}
    segs = data.get("segments")
    if not isinstance(segs, list) or not segs:
        raise ValueError(f"{path}: no 'segments' list found")
    return data


def _match_fraction(points: List[Tuple[float, float]], lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                    tol: float) -> float:
    if not points:
        return 0.0
    idx = GridIndex(20.0)
    for i, (a, b) in enumerate(lines):
        idx.insert(i, a[0], a[1], b[0], b[1])
    hit = 0
    for p in points:
        for li in idx.query(p[0] - tol, p[1] - tol, p[0] + tol, p[1] + tol):
            a, b = lines[li]
            if point_segment_distance(p, a, b)[0] <= tol:
                hit += 1
                break
    return hit / len(points)


def validate_ownership(label: str, data: dict, segs: List[dict], pdf_path: str, pdf_sha: str, page,
                       vec: dict, cfg: V6Config, frame: Optional[dict] = None) -> dict:
    """Refuse a segment JSON that was not generated from this PDF; detect its coordinate frame
    and convert every point to the analysis frame (display copies are kept for the output)."""
    src = data.get("source_pdf") or {}
    report = {"method": None, "frame": None, "geometry_match": None}
    if src.get("sha256"):
        if src["sha256"] != pdf_sha:
            raise OwnershipError(
                f"Segment JSON {label} was generated from '{src.get('file_name') or src.get('path')}' "
                f"(sha256 {src['sha256'][:12]}...), not from '{os.path.basename(pdf_path)}' "
                f"(sha256 {pdf_sha[:12]}...). Regenerate the segment JSON for this exact PDF.")
        if src.get("page") is not None and int(src["page"]) != int(page.number):
            raise OwnershipError(f"Segment JSON {label} was generated for page {src['page']}, "
                                 f"but --page is {page.number}.")
        report["method"] = "sha256"
    if frame is None:
        ident = fitz.Matrix(1, 0, 0, 1, 0, 0)
        frame = {"M": page.rotation_matrix, "R": ident, "Rinv": ident}
    H = float(page.mediabox.height)
    disp = page.rotation_matrix
    # JSON frame -> displayed page
    to_disp = {"displayed_page": lambda p: (float(p[0]), float(p[1])),
               "unrotated_page": lambda p: to_view(disp, p),
               "unrotated_y_up": lambda p: to_view(disp, (p[0], H - p[1]))}
    declared = str(data.get("coordinate_frame") or "")
    if declared == "analysis_frame" and data.get("analysis_to_display"):
        T = fitz.Matrix(*data["analysis_to_display"])
        to_disp["analysis_frame"] = lambda p: apply_matrix(T, p)
        names = ["analysis_frame"]
    elif declared == "displayed_page":
        names = ["displayed_page"]
    else:
        names = ["displayed_page", "unrotated_page", "unrotated_y_up"]
    R = frame["R"]
    pts = [p for s in segs for p in s["polyline"]]
    lines = [(l["p0"], l["p1"]) for l in vec["lines"]]
    frames = {k: _match_fraction([apply_matrix(R, to_disp[k](p)) for p in pts], lines, cfg.geometry_match_tol)
              for k in names}
    chosen = max(frames.items(), key=lambda kv: kv[1])[0]
    report["frame"] = chosen
    report["geometry_match"] = {k: rnd(v, 3) for k, v in frames.items()}
    if frames[chosen] < cfg.min_geometry_match:
        raise OwnershipError(
            f"Segment JSON {label}: only {frames[chosen] * 100:.0f}% of its polyline points lie on vector "
            f"lines of '{os.path.basename(pdf_path)}'. The JSON does not belong to this PDF (or page).")
    if report["method"] is None:
        report["method"] = "geometry"
    f = to_disp[chosen]
    for s in segs:
        s["polyline_display"] = [f(p) for p in s["polyline"]]
        s["polyline"] = [apply_matrix(R, p) for p in s["polyline_display"]]
        for ep in (s["start"], s["end"]):
            ep["point_display"], ep["node_point_display"] = f(ep["point"]), f(ep["node_point"])
            ep["point"], ep["node_point"] = apply_matrix(R, ep["point_display"]), apply_matrix(R, ep["node_point_display"])
        s["json_stations"] = [apply_matrix(R, f(p)) for p in s["json_stations"]]
    return report


# ===========================================================================
# Node grouping (which segment ends share one physical node)
# ===========================================================================

def group_nodes(segs: List[dict], tol: float = 1.5) -> Dict[str, List[Tuple[int, str]]]:
    """Assign ep['node_key'] for every segment end; return node_key -> [(seg pos, 'start'|'end')]."""
    ends = []
    for si, s in enumerate(segs):
        for side in ("start", "end"):
            ends.append((si, side, s[side]))
    parent = list(range(len(ends)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_id: Dict[str, int] = {}
    idx = GridIndex(10.0)
    for k, (_, _, ep) in enumerate(ends):
        if ep.get("node_id") is not None:
            key = str(ep["node_id"])
            if key in by_id:
                parent[find(k)] = find(by_id[key])
            else:
                by_id[key] = k
            continue
        p = ep["node_point"]
        for j in idx.query(p[0] - tol, p[1] - tol, p[0] + tol, p[1] + tol):
            if dist(p, ends[j][2]["node_point"]) <= tol:
                parent[find(k)] = find(j)
        idx.insert(k, p[0], p[1], p[0], p[1])
    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for k, (si, side, ep) in enumerate(ends):
        key = f"K{find(k)}"
        ep["node_key"] = key
        groups[key].append((si, side))
    return groups


# ===========================================================================
# Wire pieces and stations
# ===========================================================================

def detect_stations(seg: dict, vec: dict, line_index: GridIndex, circle_index: GridIndex, cfg: V6Config) -> List[float]:
    """Arc-length positions of breakpoints, leader feet and ticks on the route."""
    poly = seg["polyline"]
    L = polyline_length(poly)
    m = cfg.end_margin
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    pad = 2.0 * cfg.station_tol
    box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    found = []
    for ci in circle_index.query(*box):
        c = vec["small_circles"][ci]
        if c["filled"] and c["diameter"] > cfg.small_circle_max_diameter:
            continue              # a filled symbol, not a dot (a dot inside a route is a breakpoint)
        d, s, _ = project_on_polyline(c["center"], poly)
        if d <= cfg.station_tol and m < s < L - m:
            found.append(s)
    for li in line_index.query(*box):
        o = vec["lines"][li]
        if not o["significant"]:
            continue              # wire pen, dashed, border or texture (hatching)
        a, b = o["p0"], o["p1"]
        if dist(a, b) > cfg.station_line_max_length:
            continue
        v = unit((b[0] - a[0], b[1] - a[1]))
        for p, q in zip(poly, poly[1:]):
            u = unit((q[0] - p[0], q[1] - p[1]))
            if abs(u[0] * v[0] + u[1] * v[1]) > math.cos(math.radians(30)):
                continue
            hit = touch_or_cross(a, b, p, q, cfg.station_tol)
            if hit is not None:
                d, s, _ = project_on_polyline(hit, poly)
                if m < s < L - m:
                    found.append(s)
    for p in seg["json_stations"]:
        d, s, _ = project_on_polyline(p, poly)
        if d <= cfg.station_tol and m < s < L - m:
            found.append(s)
    found.sort()
    merged: List[float] = []
    for s in found:
        if not merged or s - merged[-1] >= cfg.station_merge_dist:
            merged.append(s)
    return merged


def build_pieces(tag: str, seg: dict, stations: List[float], cfg: V6Config) -> List[dict]:
    """Straight wire pieces: polyline edges, further divided at stations."""
    pieces = []
    poly = seg["polyline"]
    cum = 0.0
    n = 0
    for k, (a, b) in enumerate(zip(poly, poly[1:])):
        L = dist(a, b)
        if L <= 1e-9:
            continue
        cuts = [s - cum for s in stations if cum + 1e-6 < s < cum + L - 1e-6]
        bounds = [0.0] + cuts + [L]
        orient = piece_orientation(a, b, cfg.axis_tol_deg)
        for t0, t1 in zip(bounds, bounds[1:]):
            if t1 - t0 <= 1e-6:
                continue
            p0 = (a[0] + (b[0] - a[0]) * t0 / L, a[1] + (b[1] - a[1]) * t0 / L)
            p1 = (a[0] + (b[0] - a[0]) * t1 / L, a[1] + (b[1] - a[1]) * t1 / L)
            pieces.append({
                "piece_id": f"{tag}:S{seg['index']}:P{n}",
                "segment_index": seg["index"], "piece_index": n, "polyline_edge_index": k,
                "start": p0, "end": p1, "orientation": orient, "length": t1 - t0,
                "cum_start": cum + t0, "cum_end": cum + t1,
            })
            n += 1
        cum += L
    return pieces


# ===========================================================================
# Strict numeric text association with GLOBAL ownership
# ===========================================================================

def strict_label_metrics(piece: dict, text: dict, cfg: V6Config) -> Tuple[Optional[dict], str]:
    """Validate one numeric text against one straight piece using the placement conventions.

    Analysis frame: y grows downwards, so BELOW = larger y and RIGHT = larger x.  The required
    side is the calibrated convention of the PDF: side_h +1 = below a horizontal wire (the
    standard), -1 = above, 0 = printed on the wire; side_v +1 = right of a vertical wire (the
    standard), -1 = left, 0 = on the wire.
    """
    if not text["is_numeric"]:
        return None, "not_numeric"
    orient = piece["orientation"]
    if orient not in ("horizontal", "vertical"):
        return None, "angled_piece"            # angled pieces never claim numeric labels
    p0, p1 = piece["start"], piece["end"]
    cx, cy = text["center"]
    bx0, by0, bx1, by1 = text["bbox"]
    if orient == "horizontal":
        if text["orientation"] != "horizontal":
            return None, "text_not_horizontal"
        a, b = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
        lo, hi, along = a[0], b[0], cx
        if not (lo - cfg.projection_margin <= cx <= hi + cfg.projection_margin):
            return None, "outside_projection"
        t = 0.0 if hi - lo <= 1e-9 else min(1.0, max(0.0, (cx - lo) / (hi - lo)))
        wire = a[1] + t * (b[1] - a[1])
        raw = cy - wire                         # > 0 : below the wire
        conv = 1 if cfg.side_h is None else cfg.side_h
        gap = (by0 - wire) if conv >= 0 else (wire - by1)
        half_extent = (bx1 - bx0) / 2.0
    else:
        if text["orientation"] != "vertical":
            return None, "text_not_vertical"
        a, b = (p0, p1) if p0[1] <= p1[1] else (p1, p0)
        lo, hi, along = a[1], b[1], cy
        if not (lo - cfg.projection_margin <= cy <= hi + cfg.projection_margin):
            return None, "outside_projection"
        t = 0.0 if hi - lo <= 1e-9 else min(1.0, max(0.0, (cy - lo) / (hi - lo)))
        wire = a[0] + t * (b[0] - a[0])
        raw = cx - wire                         # > 0 : right of the wire
        conv = 1 if cfg.side_v is None else cfg.side_v
        gap = (bx0 - wire) if conv >= 0 else (wire - bx1)
        half_extent = (by1 - by0) / 2.0
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
        if gap < -cfg.label_overlap_tol:
            return None, "overlaps_wire"
    length = hi - lo
    mid = (lo + hi) / 2.0
    centre_err = abs(along - mid)
    half = max(length / 2.0, 0.2 * cfg.text_unit)
    projection = 0.5 if length <= 1e-9 else (along - lo) / length
    score = (cfg.weight_perpendicular * side / limit
             + cfg.weight_centre * min(centre_err / half, 2.0))
    # position along the oriented piece (start -> end)
    along_piece = (along - lo) if (p0[0] <= p1[0] if orient == "horizontal" else p0[1] <= p1[1]) else (hi - along)
    return {
        "orientation": orient, "perpendicular_distance": rnd(side), "edge_gap": rnd(gap),
        "projection": rnd(projection, 4), "centre_error": rnd(centre_err), "piece_length": rnd(length),
        "score": rnd(score, 4), "along": max(0.0, min(length, along_piece)), "half_extent": half_extent,
    }, "ok"


def assign_numeric_labels(texts: List[dict], pieces: List[dict], cfg: V6Config) -> Tuple[List[dict], dict]:
    """Global greedy assignment: every numeric text object is owned by at most one wire piece."""
    pidx = GridIndex(25.0)
    m = max(cfg.label_max_side, cfg.inline_max_side) + cfg.projection_margin + 1.0
    for i, p in enumerate(pieces):
        pidx.insert(i, min(p["start"][0], p["end"][0]) - m, min(p["start"][1], p["end"][1]) - m,
                    max(p["start"][0], p["end"][0]) + m, max(p["start"][1], p["end"][1]) + m)
    candidates = []
    rejected = Counter()
    per_text_candidates: Dict[str, int] = Counter()
    for ti, t in enumerate(texts):
        if not t["is_numeric"]:
            continue
        b = t["bbox"]
        for pi in pidx.query(b[0], b[1], b[2], b[3]):
            met, why = strict_label_metrics(pieces[pi], t, cfg)
            if met is None:
                rejected[why] += 1
                continue
            candidates.append((met["score"], met["perpendicular_distance"], met["centre_error"], t["id"], pi, ti, met))
            per_text_candidates[t["id"]] += 1
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3], pieces[c[4]]["piece_id"]))
    owner: Dict[str, int] = {}
    slots: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    alternatives: Dict[str, List[dict]] = defaultdict(list)
    assignments = []
    slot_conflicts = 0
    for score, perp, cerr, tid, pi, ti, met in candidates:
        p = pieces[pi]
        if tid in owner:
            if len(alternatives[tid]) < 3:
                alternatives[tid].append({"piece_id": p["piece_id"], "segment_index": p["segment_index"],
                                          "score": met["score"]})
            continue
        lo, hi = met["along"] - met["half_extent"], met["along"] + met["half_extent"]
        if any(lo < h + cfg.label_slot_gap and hi > l - cfg.label_slot_gap for l, h in slots[pi]):
            slot_conflicts += 1
            continue
        owner[tid] = pi
        slots[pi].append((lo, hi))
        t = texts[ti]
        assignments.append({
            "text_id": tid, "text": t["text"], "value": t["norm"], "text_center": t["center"],
            "piece_id": p["piece_id"], "segment_index": p["segment_index"], "piece_index": p["piece_index"],
            "orientation": met["orientation"], "perpendicular_distance": met["perpendicular_distance"],
            "edge_gap": met["edge_gap"], "projection": met["projection"], "centre_error": met["centre_error"],
            "piece_length": met["piece_length"], "score": met["score"],
            "s": rnd(p["cum_start"] + met["along"]),
            "confidence": rnd(max(0.05, 1.0 - 0.5 * met["score"]), 3),
            "candidate_count": per_text_candidates[tid],
        })
    for a in assignments:
        a["alternatives_not_taken"] = alternatives.get(a["text_id"], [])
    numeric_ids = {t["id"] for t in texts if t["is_numeric"]}
    unassigned_with_candidates = sorted(set(per_text_candidates) - set(owner))
    diag = {
        "numeric_texts": len(numeric_ids),
        "valid_candidates": len(candidates),
        "assigned": len(assignments),
        "unique_text_ids_assigned": len(owner),
        "unassigned_with_candidates": unassigned_with_candidates,
        "unassigned_without_candidates": len(numeric_ids) - len(per_text_candidates),
        "slot_conflicts": slot_conflicts,
        "rejection_reasons": dict(rejected),
    }
    assert len(owner) == len(assignments), "a text object was assigned more than once"
    return assignments, diag


# ===========================================================================
# Terminal / element identity with GLOBAL ownership
# ===========================================================================

def enrich_identities(segs: List[dict], groups: Dict[str, List[Tuple[int, str]]], texts: List[dict],
                      cfg: V6Config) -> dict:
    """One meaningful text object identifies at most one terminal / element node."""
    tindex = GridIndex(20.0)
    for i, t in enumerate(texts):
        tindex.insert(i, *t["bbox"])
    cands = []
    nodes = {}
    for key, members in groups.items():
        eps = [segs[si][side] for si, side in members]
        types = {ep["type"] for ep in eps}
        if not types & {"terminal", "open", "element"} or "junction" in types:
            continue
        ep0 = eps[0]
        p = ep0["node_point"] if ep0["type"] == "element" else ep0["point"]
        si, side = members[0]
        poly = segs[si]["polyline"]
        inward = unit((poly[1][0] - poly[0][0], poly[1][1] - poly[0][1])) if side == "start" else \
            unit((poly[-2][0] - poly[-1][0], poly[-2][1] - poly[-1][1]))
        is_element = "element" in types
        radius = cfg.element_text_radius if is_element else cfg.terminal_text_radius
        nodes[key] = {"point": p, "types": sorted(types), "is_element": is_element, "eps": eps}
        for ti in tindex.query(p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius):
            t = texts[ti]
            if not is_meaningful_identifier(t["norm"]):
                continue
            d = rect_distance(p, t["bbox"])
            if d > radius:
                continue
            score = d
            v = (t["center"][0] - p[0], t["center"][1] - p[1])
            up = (t["dir"][1], -t["dir"][0])                 # "up" in the text's own reading frame
            below = -(v[0] * up[0] + v[1] * up[1])
            if below > 0:                                    # prefer text above / level with the end
                score += cfg.terminal_below_penalty * min(1.0, below / (2.0 * cfg.terminal_below_slack))
            # names are printed level with their terminal: small cost for being off that line
            score += cfg.terminal_align_weight * max(0.0, abs(below) - 0.5 * cfg.text_unit)
            if not is_element and v[0] * inward[0] + v[1] * inward[1] > 0.67 * cfg.terminal_below_slack:
                score += cfg.terminal_inward_penalty         # text lying over the wire side
            cands.append((score, d, key, ti))
    cands.sort(key=lambda c: (c[0], c[1], c[2], texts[c[3]]["id"]))
    used_text, used_node = set(), set()
    chosen = {}
    for score, d, key, ti in cands:
        if key in used_node or ti in used_text:
            continue
        r = cfg.element_text_radius if nodes[key]["is_element"] else cfg.terminal_text_radius
        conf = max(0.1, min(1.0, 1.0 - score / (2.0 * r)))
        if conf < cfg.terminal_min_confidence:
            continue
        used_node.add(key)
        used_text.add(ti)
        chosen[key] = (texts[ti], conf, score)
    # weak fallback (e.g. a BRAID box) for nodes without a meaningful identifier
    for key, nd in nodes.items():
        if key in chosen or nd["is_element"]:
            continue
        p = nd["point"]
        best = None
        rw = 0.4 * cfg.terminal_text_radius
        for ti in tindex.query(p[0] - rw, p[1] - rw, p[0] + rw, p[1] + rw):
            t = texts[ti]
            if ti in used_text or annotation_keyword(t["norm"]) != "BRAID":
                continue
            d = rect_distance(p, t["bbox"])
            if d <= rw and (best is None or d < best[0]):
                best = (d, ti)
        if best:
            used_text.add(best[1])
            chosen[key] = (texts[best[1]], 0.3, best[0])
    report = {"nodes": 0, "named": 0, "changed_from_original": 0, "unidentified": 0}
    chosen_names = {canonical_name(v[0]["norm"]) for v in chosen.values()}
    for key, nd in nodes.items():
        report["nodes"] += 1
        hit = chosen.get(key)
        for ep in nd["eps"]:
            orig = ep["original_structural_id"]
            if hit:
                t, conf, score = hit
                name = canonical_name(t["norm"])
                ep["name"] = name
                ep["name_confidence"] = rnd(conf, 3)
                ep["identity"] = {"name": name, "text_id": t["id"], "text": t["text"], "confidence": rnd(conf, 3),
                                  "score": rnd(score, 3), "method": "global_terminal_text"}
                if ep["type"] == "open" and conf >= 0.5:
                    ep["type"] = "terminal"
                if ep["type"] == "terminal":
                    ep["structural_id"] = f"TERMINAL[{name}]"
                elif ep["type"] == "element":
                    ep["structural_id"] = f"ELEMENT[{name}]"
            elif ep.get("name") and is_meaningful_identifier(ep["name"]) and ep["name"] not in chosen_names:
                # keep a meaningful original identity that no other endpoint owns
                ep["identity"] = {"name": ep["name"], "text_id": None, "confidence": 0.4, "method": "original_json"}
                ep["name_confidence"] = 0.4
            else:
                ep["identity"] = {"name": None, "text_id": None, "confidence": 0.0, "method": "none"}
                ep["name_confidence"] = 0.0
                if ep["type"] == "terminal":
                    ep["name"] = None
                    ep["structural_id"] = "TERMINAL[UNIDENTIFIED]"
                elif ep["type"] == "element":
                    ep["name"] = None
                    ep["structural_id"] = "ELEMENT[UNNAMED]"
            if ep["structural_id"] != orig:
                report["changed_from_original"] += 1
        report["named" if hit else "unidentified"] += 1
    used_ids = [chosen[k][0]["id"] for k in chosen]
    assert len(used_ids) == len(set(used_ids)), "a terminal text object identifies more than one endpoint"
    return report


# ===========================================================================
# Sequence rebuild, junction ID rebuild, segment keys
# ===========================================================================

def estimate_missing_positions(seg: dict, texts: List[dict], tindex: GridIndex, window: float = 25.0) -> None:
    """Original ELEMENT items without a stored position: project their annotation text on the route."""
    used = set()
    poly = seg["polyline"]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    for it in seg["orig_items"]:
        if it["s"] is not None or token_kind(it["token"]) != "ELEMENT":
            continue
        val = token_value(it["token"])
        kw = annotation_keyword(val)
        best = None
        for ti in tindex.query(min(xs) - window, min(ys) - window, max(xs) + window, max(ys) + window):
            t = texts[ti]
            if ti in used:
                continue
            if not (t["norm"] == val or (kw and annotation_keyword(t["norm"]) == kw)):
                continue
            d, s, _ = project_on_polyline(t["center"], poly)
            if d <= window and (best is None or d < best[0]):
                best = (d, s, ti)
        if best:
            used.add(best[2])
            it["s"] = best[1]
            it["s_estimated"] = True


def rebuild_sequences(segs: List[dict], assignments: List[dict], texts: List[dict],
                      window: float = 25.0) -> None:
    by_seg: Dict[int, List[dict]] = defaultdict(list)
    for a in assignments:
        by_seg[a["segment_index"]].append(a)
    tindex = GridIndex(20.0)
    for i, t in enumerate(texts):
        tindex.insert(i, *t["bbox"])
    for s in segs:
        estimate_missing_positions(s, texts, tindex, window)
        items = []
        for a in by_seg.get(s["index"], []):
            items.append({"kind": "WIRE", "value": a["value"], "token": f"WIRE:{a['value']}", "s": a["s"],
                          "text_id": a["text_id"], "piece_id": a["piece_id"], "score": a["score"]})
        wires_s = sorted(it["s"] for it in items)
        last_s = 0.0
        for k, it in enumerate(s["orig_items"]):
            if token_kind(it["token"]) != "ELEMENT":
                if it["s"] is not None:
                    last_s = it["s"]
                continue
            pos = it["s"]
            if pos is None:                       # keep the original relative order as well as possible
                later = [w for w in wires_s if w > last_s]
                pos = (last_s + later[0]) / 2.0 if later else last_s + 0.5
            items.append({"kind": "ELEMENT", "value": token_value(it["token"]), "token": it["token"],
                          "s": rnd(pos), "annotation": it["annotation"], "caption": it.get("caption"),
                          "source": "original_items"})
            last_s = pos
        items.sort(key=lambda x: (x["s"], 0 if x["kind"] == "WIRE" else 1))
        s["reb_items"] = items
        s["reb_sequence"] = [x["token"] for x in items]
        s["reb_wires"] = [x["value"] for x in items if x["kind"] == "WIRE"]
        s["assignments"] = sorted(by_seg.get(s["index"], []), key=lambda a: a["s"])


def rebuild_junction_ids(segs: List[dict], groups: Dict[str, List[Tuple[int, str]]]) -> None:
    """Depth-1 label of each direction = first WIRE value from that end of the rebuilt sequence."""
    for key, members in groups.items():
        eps = [segs[si][side] for si, side in members]
        if not any(ep["type"] == "junction" for ep in eps):
            continue
        labels = []
        for si, side in members:
            w = segs[si]["reb_wires"]
            labels.append((w[0] if side == "start" else w[-1]) if w else UNLABELLED)
        prefix = next((ep["prefix"] for ep in eps if ep.get("prefix")), None) or "DOT1"
        rid = junction_id(prefix, labels)
        for ep in eps:
            ep["rebuilt_labels"] = labels
            ep["rebuilt_structural_id"] = rid
            if ep.get("labels") is None:
                ep["labels"] = list(labels)


def segment_key(a: Optional[str], b: Optional[str]) -> str:
    x, y = sorted([a or "?", b or "?"])
    return f"SEG[{x}<->{y}]"


def final_id(ep: dict) -> Optional[str]:
    if ep["type"] == "junction":
        return ep.get("rebuilt_structural_id") or ep["structural_id"]
    return ep["structural_id"]


# ===========================================================================
# Similarity measures
# ===========================================================================

_MS_CACHE: Dict[tuple, Tuple[float, dict]] = {}


def multiset_junction_similarity(la: Optional[List[str]], lb: Optional[List[str]], min_common: int) -> Tuple[float, dict]:
    """Junction depth-1 IDs compared as multisets; UNLABELLED is a wildcard with half credit (cached)."""
    key = (tuple(la) if la else None, tuple(lb) if lb else None, min_common)
    hit = _MS_CACHE.get(key)
    if hit is None:
        hit = _multiset_junction_similarity(la, lb, min_common)
        _MS_CACHE[key] = hit
    return hit[0], dict(hit[1])


def _multiset_junction_similarity(la, lb, min_common: int) -> Tuple[float, dict]:
    if not la or not lb:
        return 0.3, {"common": 0, "related": False}
    A = Counter(x for x in la if x != UNLABELLED)
    B = Counter(x for x in lb if x != UNLABELLED)
    ua, ub = sum(1 for x in la if x == UNLABELLED), sum(1 for x in lb if x == UNLABELLED)
    common = sum((A & B).values())
    rem_a, rem_b = sum(A.values()) - common, sum(B.values()) - common
    w1, w2 = min(rem_a, ub), min(rem_b, ua)
    w3 = min(ua - w2, ub - w1)
    n = max(len(la), len(lb))
    raw = (common + 0.5 * (w1 + w2 + w3)) / n
    need = min(min_common, sum(A.values()), sum(B.values()))
    related = common >= max(1, need) if (A and B) else (w1 + w2 + w3) > 0
    exact = Counter(la) == Counter(lb)
    info = {"common": common, "unresolved": w1 + w2 + w3, "related": related, "exact": exact,
            "only_a": sorted((A - B).elements(), key=numeric_sort_key),
            "only_b": sorted((B - A).elements(), key=numeric_sort_key)}
    if exact:
        return 1.0, info
    return (raw if related else min(raw, 0.25)), info


def remove_one(labels: Optional[List[str]], value: Optional[str]) -> Optional[List[str]]:
    """Junction labels without the label of the segment's own direction."""
    if labels is None:
        return None
    out = list(labels)
    target = value if value is not None else UNLABELLED
    if target in out:
        out.remove(target)
    return out


def endpoint_similarity(ea: dict, eb: dict, cfg: V6Config, ctx: Optional[dict] = None,
                        own_a: Tuple[Optional[str], Optional[str]] = (None, None),
                        own_b: Tuple[Optional[str], Optional[str]] = (None, None)) -> Tuple[float, bool, List[str]]:
    """(similarity, exact, notes) for an endpoint of a PDF A segment and one of a PDF B segment.

    own_a / own_b = (original, rebuilt) label of the segment's own direction at this endpoint.
    A junction is compared on its full depth-1 multiset, on its context (the other
    directions only, so a changed label of this very segment does not hide the junction),
    and through the connectivity-based node correspondence when available.
    """
    ta, tb = ea["type"], eb["type"]
    notes: List[str] = []
    if ta == "junction" and tb == "junction":
        best, best_info = -1.0, None
        sources = ((ea.get("labels"), eb.get("labels"), own_a[0], own_b[0]),
                   (ea.get("rebuilt_labels"), eb.get("rebuilt_labels"), own_a[1], own_b[1]))
        for la, lb, oa, ob in sources:
            if la is None or lb is None:
                continue
            sim, info = multiset_junction_similarity(la, lb, cfg.junction_min_common)
            if sim > best:
                best, best_info = sim, info
            ca, cb = remove_one(la, oa), remove_one(lb, ob)
            if ca and cb and (len(ca) < len(la) or len(cb) < len(lb)):
                csim, cinfo = multiset_junction_similarity(ca, cb, cfg.junction_min_common)
                if csim > best:
                    best, best_info = csim, dict(cinfo, context=True)
        if best < 0:
            best, best_info = 0.3, {}
            notes.append("junction labels unavailable")
        exact = bool(best_info.get("exact") and not best_info.get("context")) or (final_id(ea) == final_id(eb)) or \
            (ea["original_structural_id"] == eb["original_structural_id"])
        corr = (ctx or {}).get("a2b", {}).get(ea.get("node_key"))
        if not exact and corr and corr[0] == eb.get("node_key") and corr[1] > best:
            best = corr[1]
            notes.append(f"junctions correspond through connectivity (neighbour support {corr[2]:.2f})")
        if not exact:
            if best_info.get("context"):
                notes.append("junction matched on its other directions (this segment's own label differs)")
            if best_info.get("unresolved"):
                notes.append(f"junction has {best_info['unresolved']} unresolved (UNLABELLED) label(s)")
            if best_info.get("only_a") or best_info.get("only_b"):
                notes.append(f"junction depth-1 signature differs (A only {best_info['only_a']}, "
                             f"B only {best_info['only_b']})")
        return (1.0 if exact else best), exact, notes
    if ta == "open" and tb == "open":
        exact = ea["structural_id"] == eb["structural_id"]
        if not exact:
            notes.append("OPEN cluster index differs (unstable, ignored)")
        return (1.0 if exact else 0.85), exact, notes
    if ta in ("terminal", "element") and tb == ta:
        na, nb = ea.get("name"), eb.get("name")
        if na and nb and na == nb:
            return 1.0, True, notes
        ca, cb = float(ea.get("name_confidence") or 0), float(eb.get("name_confidence") or 0)
        if not na and not nb:
            notes.append(f"{ta} identity unresolved on both sides")
            return 0.5, False, notes
        if not na or not nb:
            notes.append(f"{ta} identity unresolved on one side")
            return 0.45, False, notes
        if min(ca, cb) < 0.5:
            notes.append(f"uncertain {ta} identity ({na} vs {nb})")
            return 0.4, False, notes
        notes.append(f"{ta} identity differs ({na} vs {nb})")
        return 0.1, False, notes
    pair = {ta, tb}
    if pair == {"open", "terminal"}:
        notes.append("open endpoint vs terminal")
        return 0.45, False, notes
    if pair == {"element", "terminal"}:
        notes.append("element vs terminal endpoint")
        return 0.35, False, notes
    if "junction" in pair:
        return 0.1, False, ["junction vs non-junction endpoint"]
    return 0.3, False, [f"{ta} vs {tb} endpoint"]


def token_cost(x: str, y: str) -> int:
    if x == y:
        return 0
    return 1 if token_kind(x) == token_kind(y) else 2


def edit_matrix(a: Sequence[str], b: Sequence[str]) -> List[List[int]]:
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + token_cost(a[i - 1], b[j - 1]))
    return D


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    return edit_matrix(a, b)[-1][-1]


_ALIGN_CACHE: Dict[tuple, Tuple[float, bool, int]] = {}


def align(a: Sequence[str], b: Sequence[str]) -> Tuple[float, bool, int]:
    """Best of direct and reversed order: (similarity, reversed, distance) (cached)."""
    key = (tuple(a), tuple(b))
    hit = _ALIGN_CACHE.get(key)
    if hit is None:
        hit = _align(a, b)
        _ALIGN_CACHE[key] = hit
    return hit


def _align(a: Sequence[str], b: Sequence[str]) -> Tuple[float, bool, int]:
    d_fwd = edit_distance(a, b)
    d_rev = edit_distance(a, list(reversed(b)))
    rev = d_rev < d_fwd
    d = d_rev if rev else d_fwd
    n = max(len(a), len(b))
    return (1.0 if n == 0 else max(0.0, 1.0 - d / n)), rev, d


def edit_script(a: Sequence[str], b: Sequence[str]) -> List[dict]:
    D = edit_matrix(a, b)
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + token_cost(a[i - 1], b[j - 1]) \
                and token_cost(a[i - 1], b[j - 1]) <= 1:
            op = "equal" if a[i - 1] == b[j - 1] else "substitute"
            ops.append({"op": op, "a_index": i - 1, "b_index": j - 1, "a_token": a[i - 1], "b_token": b[j - 1]})
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append({"op": "delete", "a_index": i - 1, "b_index": None, "a_token": a[i - 1], "b_token": None})
            i -= 1
        else:
            ops.append({"op": "insert", "a_index": None, "b_index": j - 1, "a_token": None, "b_token": b[j - 1]})
            j -= 1
    ops.reverse()
    return ops


def wire_similarity(wa: Sequence[str], wb: Sequence[str]) -> float:
    A, B = Counter(wa), Counter(wb)
    union = sum((A | B).values())
    return 1.0 if union == 0 else sum((A & B).values()) / union


# ===========================================================================
# Pair scoring and matching
# ===========================================================================

def first_wire(seq: Sequence[str], from_end: bool) -> Optional[str]:
    for tok in (reversed(seq) if from_end else seq):
        if token_kind(tok) == "WIRE":
            return token_value(tok)
    return None


def own_labels(seg: dict, side: str) -> Tuple[Optional[str], Optional[str]]:
    """(original, rebuilt) depth-1 label of the segment's own direction at one end."""
    e = side == "end"
    return first_wire(seg["orig_sequence"], e), first_wire(seg["reb_sequence"], e)


def endpoint_alignment(sa: dict, sb: dict, cfg: V6Config, ctx: Optional[dict] = None) -> dict:
    oas, oae, obs, obe = own_labels(sa, "start"), own_labels(sa, "end"), own_labels(sb, "start"), own_labels(sb, "end")
    d1, e1, n1 = endpoint_similarity(sa["start"], sb["start"], cfg, ctx, oas, obs)
    d2, e2, n2 = endpoint_similarity(sa["end"], sb["end"], cfg, ctx, oae, obe)
    r1, f1, m1 = endpoint_similarity(sa["start"], sb["end"], cfg, ctx, oas, obe)
    r2, f2, m2 = endpoint_similarity(sa["end"], sb["start"], cfg, ctx, oae, obs)
    direct, swapped = (d1 + d2) / 2, (r1 + r2) / 2
    if swapped > direct + 1e-9:
        return {"mean": swapped, "min": min(r1, r2), "max": max(r1, r2), "reversed": True,
                "exact": f1 and f2, "notes": m1 + m2, "pair": (r1, r2)}
    return {"mean": direct, "min": min(d1, d2), "max": max(d1, d2), "reversed": False,
            "exact": e1 and e2, "notes": n1 + n2, "pair": (d1, d2)}


def score_pair(sa: dict, sb: dict, cfg: V6Config, ctx: Optional[dict] = None) -> Optional[dict]:
    ep = endpoint_alignment(sa, sb, cfg, ctx)
    c_reb, rev_reb, _ = align(sa["reb_sequence"], sb["reb_sequence"])
    c_orig, rev_orig, _ = align(sa["orig_sequence"], sb["orig_sequence"])
    content = max(c_reb, c_orig)
    wsim = max(wire_similarity(sa["reb_wires"], sb["reb_wires"]), wire_similarity(sa["orig_wires"], sb["orig_wires"]))
    total = cfg.weight_endpoint * ep["mean"] + cfg.weight_content * content + cfg.weight_wire * wsim
    n_tok = max(len(sa["reb_sequence"]), len(sb["reb_sequence"]))
    strong = ep["min"] >= cfg.strong_endpoint_min and ep["mean"] >= cfg.strong_endpoint_mean
    partial = ep["max"] >= cfg.partial_endpoint_max and content >= cfg.partial_content_min
    content_driven = content >= cfg.content_driven_min and n_tok >= 2 and ep["mean"] >= 0.3
    if not (strong or partial or content_driven) or total < cfg.min_pair_score:
        return None
    return {"score": total, "endpoint": ep, "content": content, "wire": wsim,
            "gate": "endpoints" if strong else ("endpoint+content" if partial else "content")}


def candidate_pairs(sa_list: List[dict], sb_list: List[dict], a2b: dict) -> List[Tuple[int, int]]:
    """Segment pairs worth scoring: they share a wire value, an endpoint name, a junction label,
    a corresponding node, or (for segments without wire numbers) their endpoint type pair."""
    b_mapped = {v[0] for v in a2b.values()}

    def keys(seg: dict, node_map, mapped) -> set:
        ks = set()
        for w in set(seg["reb_wires"]) | set(seg["orig_wires"]):
            ks.add(("W", w))
        for ep in (seg["start"], seg["end"]):
            if ep["type"] in ("terminal", "element") and ep.get("name"):
                ks.add(("N", ep["name"]))
            k = node_map(ep.get("node_key"))
            if k is not None:
                ks.add(("K", k))
            if ep["type"] == "junction" and not mapped(ep.get("node_key")):
                for l in set((ep.get("labels") or []) + (ep.get("rebuilt_labels") or [])):
                    if l != UNLABELLED:
                        ks.add(("J", l))
        if not seg["reb_wires"] and not seg["orig_wires"]:
            ks.add(("T",) + tuple(sorted((seg["start"]["type"], seg["end"]["type"]))))
        return ks

    index: Dict[tuple, List[int]] = defaultdict(list)
    for j, sb in enumerate(sb_list):
        for k in keys(sb, lambda nk: nk, lambda nk: nk in b_mapped):
            index[k].append(j)
    out = []
    for i, sa in enumerate(sa_list):
        js = set()
        for k in keys(sa, lambda nk: (a2b.get(nk) or (None,))[0], lambda nk: nk in a2b):
            js.update(index.get(k, ()))
        out.extend((i, j) for j in sorted(js))
    return out


def oriented(seq: List[str], toward_end: bool) -> List[str]:
    return list(seq) if toward_end else list(reversed(seq))


# ===========================================================================
# Classification of a matched pair
# ===========================================================================

def visible_near(value: str, seg: dict, texts: List[dict], tindex: GridIndex, radius: float) -> bool:
    """Is a numeric text with this value printed close to the segment (assigned elsewhere or not)?"""
    own = {a["text_id"] for a in seg.get("assignments", [])}
    poly = seg["polyline"]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    for ti in tindex.query(min(xs) - radius, min(ys) - radius, max(xs) + radius, max(ys) + radius):
        t = texts[ti]
        if t["norm"] != value or t["id"] in own:
            continue
        if project_on_polyline(t["center"], poly)[0] <= radius:
            return True
    return False


def classify_pair(sa: dict, sb: dict, pair: dict, ctx: dict, cfg: V6Config) -> dict:
    ep = pair["endpoint"]
    oa, ob = sa["orig_sequence"], sb["orig_sequence"]
    ra, rb = sa["reb_sequence"], sb["reb_sequence"]
    _, rev_r, d_r = align(ra, rb)
    _, rev_o, d_o = align(oa, ob)
    originals_present = bool(oa) or bool(ob)
    notes = list(ep["notes"])
    ops: List[dict] = []
    if d_r == 0:
        content_source, content_reversed, identical = "rebuilt_sequences", rev_r, True
        reason = "Rebuilt sequences are identical."
    elif d_o == 0 and originals_present:
        content_source, content_reversed, identical = "original_sequences_identical", rev_o, True
        reason = "The original extracted sequences are identical. A rebuilt extraction difference was ignored."
    else:
        content_source, content_reversed, identical = "rebuilt_sequences", rev_r, False
        reason = None
        rb_oriented = list(reversed(rb)) if rev_r else list(rb)
        ops = edit_script(ra, rb_oriented)
        if rev_r:                                   # report B positions in B's own order
            for op in ops:
                if op["b_index"] is not None:
                    op["b_index"] = len(rb) - 1 - op["b_index"]
    content_sim = align(ra, rb)[0] if content_source == "rebuilt_sequences" else 1.0

    # --- judge every edit operation -------------------------------------------------
    ca_o, cb_o, ca_r, cb_r = Counter(oa), Counter(ob), Counter(ra), Counter(rb)
    real, annot, uncertain = [], [], []
    script = []
    for k, op in enumerate(o for o in ops if o["op"] != "equal"):
        toks = [t for t in (op["a_token"], op["b_token"]) if t]
        if all(is_annotation_token(t) for t in toks):
            category, flag = "annotation", "annotation"
        elif any(token_kind(t) == "WIRE" for t in toks):
            category = "wire"
            unsure = []
            if op["a_token"] and token_kind(op["a_token"]) == "WIRE":
                x = op["a_token"]
                v = token_value(x)
                if cb_o[x] > cb_r[x] or ca_r[x] > ca_o[x] or \
                        visible_near(v, sb, ctx["texts_b"], ctx["tindex_b"], ctx.get("radius_b", cfg.uncertain_text_radius)):
                    unsure.append(f"{x} is still present / visible in PDF B near this segment")
            if op["b_token"] and token_kind(op["b_token"]) == "WIRE":
                y = op["b_token"]
                v = token_value(y)
                if ca_o[y] > ca_r[y] or cb_r[y] > cb_o[y] or \
                        visible_near(v, sa, ctx["texts_a"], ctx["tindex_a"], ctx.get("radius_a", cfg.uncertain_text_radius)):
                    unsure.append(f"{y} is also present / visible in PDF A near this segment")
            flag = "uncertain" if unsure else "real"
        else:
            category, flag = "component", "real"
        if op["op"] == "substitute":
            desc = f"{op['a_token']} changed to {op['b_token']}"
        elif op["op"] == "delete":
            desc = f"{op['a_token']} removed (A position {op['a_index']})"
        else:
            desc = f"{op['b_token']} added (B position {op['b_index']})"
        entry = dict(op, category=category, uncertain=(flag != "real"), description=desc)
        if flag == "uncertain":
            entry["uncertainty"] = unsure
        script.append(entry)
        {"real": real, "annotation": annot, "uncertain": uncertain}[flag].append(entry)

    # --- terminal / element identity changes at the segment ends ----------------------
    # A confident, same-type name change is a "changed". A differing identity that is only
    # low-confidence, or whose symbol / node type also changed (box terminal -> dashed symbol,
    # terminal <-> element), is surfaced as a review rather than hidden inside "same".
    endpoint_changes, endpoint_reviews = [], []
    ends_a = (sa["start"], sa["end"])
    ends_b = (sb["end"], sb["start"]) if ep["reversed"] else (sb["start"], sb["end"])
    for ea, eb in zip(ends_a, ends_b):
        na, nb = ea.get("name"), eb.get("name")
        if not (na and nb and na != nb and is_meaningful_identifier(na) and is_meaningful_identifier(nb)):
            continue
        ca, cb = float(ea.get("name_confidence") or 0), float(eb.get("name_confidence") or 0)
        same_type = ea["type"] == eb["type"] and ea["type"] in ("terminal", "element")
        if same_type and min(ca, cb) >= cfg.terminal_change_min_confidence:
            endpoint_changes.append({"category": ea["type"], "a_token": na, "b_token": nb,
                                     "description": f"{ea['type']} {na} changed to {nb}"})
        else:
            why = "symbol / node type also differs" if not same_type else "low-confidence identification"
            endpoint_reviews.append({"category": "endpoint_identity", "a_token": na, "b_token": nb,
                                     "a_type": ea["type"], "b_type": eb["type"],
                                     "description": f"endpoint identity may have changed: {na} vs {nb} ({why})"})
    # --- component symbol orientation (a connector drawn mirrored) --------------------
    # Connectivity is unchanged, but an asymmetric component drawn as its mirror image is a
    # real, meaningful change (pin/keying orientation). Handedness is traversal-invariant, so
    # the sign is compared directly by component name.
    orientation_changes = []
    thr = cfg.component_orientation_min_strength

    def comp_orient(seg: dict) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for it in seg["orig_items"]:
            o = it.get("orientation")
            if o is not None and token_kind(it["token"]) == "ELEMENT":
                v = token_value(it["token"])
                if abs(o) >= abs(m.get(v, 0.0)):
                    m[v] = o
        return m

    oa, ob = comp_orient(sa), comp_orient(sb)
    for name in sorted(set(oa) & set(ob)):
        da, db = oa[name], ob[name]
        if abs(da) >= thr and abs(db) >= thr and da * db < 0:
            orientation_changes.append({"category": "component", "token": name,
                                        "description": f"component {name} is mirrored (symbol orientation reversed)"})

    if identical and not endpoint_changes and not endpoint_reviews and not orientation_changes:
        cls = "same_exact" if ep["exact"] else "same_probable"
        if cls == "same_probable" and not notes:
            notes.append("endpoint identities differ slightly")
    elif real or endpoint_changes or orientation_changes:
        cls = "changed"
        reason = "Meaningful wire number / component / endpoint identity change."
    elif endpoint_reviews:
        cls = "review"
        reason = "An endpoint terminal / element identity may have changed; confirm it."
    elif annot and not uncertain:
        cls, reason = "review", "Only text-derived annotations differ (not a confirmed physical change)."
    else:
        cls, reason = "review", "Differences are probably extraction / association uncertainty."
    if cls == "review":
        review_type = ("endpoint_identity" if endpoint_reviews and not (annot and not uncertain)
                       else "annotation_only" if annot and not uncertain else "extraction_uncertainty")
    else:
        review_type = None
    return {
        "classification": cls, "review_type": review_type,
        "confidence": rnd(pair["score"], 4), "endpoint_confidence": rnd(ep["mean"], 4),
        "endpoint_reversed": ep["reversed"], "endpoint_exact": ep["exact"],
        "content_similarity": rnd(content_sim, 4), "match_content_similarity": rnd(pair["content"], 4),
        "wire_similarity": rnd(pair["wire"], 4), "match_gate": pair["gate"],
        "content_source": content_source, "content_reversed": content_reversed,
        "edit_script": script, "endpoint_changes": endpoint_changes, "endpoint_reviews": endpoint_reviews,
        "orientation_changes": orientation_changes,
        "changes": [e["description"] for e in script if not e["uncertain"] and e["category"] != "annotation"]
        + [e["description"] for e in endpoint_changes] + [e["description"] for e in orientation_changes],
        "annotation_changes": [e["description"] for e in script if e["category"] == "annotation"],
        "uncertain_changes": [e["description"] for e in script if e["uncertain"] and e["category"] != "annotation"]
        + [e["description"] for e in endpoint_reviews],
        "reason": reason, "notes": notes,
    }


# ===========================================================================
# Topology recovery: route merged / split at a removed or added junction
# ===========================================================================

def topology_recovery(segs_x: List[dict], groups_x: Dict[str, List[Tuple[int, str]]], free_x: set,
                      segs_y: List[dict], free_y: set, cfg: V6Config, x_is_a: bool,
                      ctx: Optional[dict] = None) -> List[dict]:
    """Two unmatched X segments joined at one X node versus one unmatched Y segment."""
    def esim(ex: dict, ey: dict) -> float:
        return (endpoint_similarity(ex, ey, cfg, ctx) if x_is_a else endpoint_similarity(ey, ex, cfg, ctx))[0]

    found = []
    for key, members in groups_x.items():
        mem = [(si, side) for si, side in members if si in free_x]
        if len(mem) < 2:
            continue
        for i in range(len(mem)):
            for j in range(len(mem)):
                if i == j or mem[i][0] == mem[j][0]:
                    continue
                (s1, side1), (s2, side2) = mem[i], mem[j]
                a, b = segs_x[s1], segs_x[s2]
                seq = oriented(a["reb_sequence"], side1 == "end") + oriented(b["reb_sequence"], side2 == "start")
                far1 = a["start"] if side1 == "end" else a["end"]
                far2 = b["end"] if side2 == "start" else b["start"]
                wires = {token_value(t) for t in seq if token_kind(t) == "WIRE"}
                for yi in free_y:
                    y = segs_y[yi]
                    if wires and not wires & set(y["reb_wires"]):
                        continue
                    c, rev, _ = align(seq, y["reb_sequence"])
                    if c < cfg.topology_content_min or len(seq) == 0:
                        continue
                    d = (esim(far1, y["start"]) + esim(far2, y["end"])) / 2
                    r = (esim(far1, y["end"]) + esim(far2, y["start"])) / 2
                    e = max(d, r)
                    if e < cfg.topology_endpoint_min:
                        continue
                    found.append({"score": 0.6 * e + 0.4 * c, "x": (s1, s2), "y": yi, "node_key": key,
                                  "content": c, "endpoint": e, "content_reversed": rev})
    found.sort(key=lambda f: (-f["score"], f["x"], f["y"]))
    used_x, used_y, out = set(), set(), []
    for f in found:
        if f["x"][0] in used_x or f["x"][1] in used_x or f["y"] in used_y:
            continue
        used_x.update(f["x"])
        used_y.add(f["y"])
        node_ep = next(segs_x[si][side] for si, side in groups_x[f["node_key"]] if si == f["x"][0])
        f["node_id"] = final_id(node_ep)
        f["x_is_a"] = x_is_a
        out.append(f)
    return out


# ===========================================================================
# Connectivity-based node correspondence and junction report
# ===========================================================================

def build_node_table(R: dict) -> Dict[str, dict]:
    """One record per physical boundary node with its neighbours along segments."""
    segs, groups = R["segs"], R["groups"]
    table = {}
    for key, members in groups.items():
        eps = [segs[si][side] for si, side in members]
        types = {e["type"] for e in eps}
        t = next((x for x in ("junction", "element", "terminal", "open") if x in types), "other")
        e0 = next((e for e in eps if e["type"] == t), eps[0])
        nbrs = []
        for si, side in members:
            nbrs.append((segs[si]["end" if side == "start" else "start"]["node_key"], si))
        table[key] = {"key": key, "type": t, "name": e0.get("name"), "labels": e0.get("labels"),
                      "rlabels": e0.get("rebuilt_labels"), "nbrs": nbrs, "id": final_id(e0),
                      "original_id": e0["original_structural_id"],
                      "point": rpt(e0.get("node_point_display") or e0["node_point"])}
    return table


def base_node_similarity(a: dict, b: dict, cfg: V6Config) -> Tuple[float, dict]:
    if a["type"] != b["type"]:
        return 0.0, {}
    if a["type"] in ("terminal", "element"):
        return (1.0 if a["name"] and a["name"] == b["name"] else 0.0), {}
    if a["type"] == "junction":
        best, binfo = 0.0, {}
        for la, lb in ((a["labels"], b["labels"]), (a["rlabels"], b["rlabels"])):
            if la and lb:
                sim, info = multiset_junction_similarity(la, lb, cfg.junction_min_common)
                if info.get("related") and sim > best:
                    best, binfo = sim, info
        return best, binfo
    return 0.0, {}


def node_correspondence(na: Dict[str, dict], nb: Dict[str, dict], cfg: V6Config):
    """One-to-one node correspondence from identities, then neighbour (connectivity) support.

    Terminals / elements anchor by name, junctions by their depth-1 multiset.  A junction
    whose own label set changed (e.g. two adjacent lengths changed) is still recovered when
    most of its neighbours correspond.  Absolute positions are never used.
    """
    by_name, by_label = defaultdict(list), defaultdict(set)
    for k, n in nb.items():
        if n["type"] in ("terminal", "element") and n["name"]:
            by_name[(n["type"], n["name"])].append(k)
        if n["type"] == "junction":
            for l in set((n["labels"] or []) + (n["rlabels"] or [])):
                if l != UNLABELLED:
                    by_label[l].add(k)
    S0, info0 = {}, {}
    for ka, a in na.items():
        cands = set()
        if a["type"] in ("terminal", "element") and a["name"]:
            cands.update(by_name.get((a["type"], a["name"]), []))
        if a["type"] == "junction":
            for l in set((a["labels"] or []) + (a["rlabels"] or [])):
                cands |= by_label.get(l, set())
        for kb in cands:
            sim, info = base_node_similarity(a, nb[kb], cfg)
            if sim > 0:
                S0[(ka, kb)] = sim
                info0[(ka, kb)] = info
    S, support = dict(S0), {}
    for _ in range(2):
        anchors = {pair: v for pair, v in S.items() if v >= cfg.node_anchor_min}
        pairs = set(S)
        for ka, kb in anchors:
            for xa, _ in na[ka]["nbrs"]:
                for xb, _ in nb[kb]["nbrs"]:
                    if xa in na and xb in nb and na[xa]["type"] == nb[xb]["type"]:
                        pairs.add((xa, xb))
        newS = {}
        for ka, kb in pairs:
            a, b = na[ka], nb[kb]
            if a["type"] != b["type"]:
                continue
            ranked = sorted(((anchors.get((xa, xb), 0.0), xa, xb) for xa, _ in a["nbrs"] for xb, _ in b["nbrs"]),
                            reverse=True)
            ua, ub, cnt = set(), set(), 0
            for v, xa, xb in ranked:
                if v <= 0:
                    break
                if xa in ua or xb in ub:
                    continue
                ua.add(xa)
                ub.add(xb)
                cnt += 1
            sup = cnt / max(len(a["nbrs"]), len(b["nbrs"]), 1)
            support[(ka, kb)] = sup
            boost = 0.9 * sup if (sup >= 0.5 and min(len(a["nbrs"]), len(b["nbrs"])) >= 2) else 0.0
            val = max(S0.get((ka, kb), 0.0), boost)
            if val > 0:
                newS[(ka, kb)] = val
        S = newS
    order = sorted(S.items(), key=lambda kv: (-(kv[1] + 0.05 * support.get(kv[0], 0.0)), kv[0]))
    a2b, b2a = {}, {}
    for (ka, kb), v in order:
        if v < cfg.node_corr_min or ka in a2b or kb in b2a:
            continue
        a2b[ka] = (kb, rnd(v, 4), rnd(support.get((ka, kb), 0.0), 3), info0.get((ka, kb), {}))
        b2a[kb] = ka
    return a2b, b2a


def junction_report(na: Dict[str, dict], nb: Dict[str, dict], a2b: dict, cfg: V6Config) -> dict:
    def pub(n):
        return {"node_key": n["key"], "original_id": n["original_id"], "rebuilt_id": n["id"],
                "labels": n["labels"], "rebuilt_labels": n["rlabels"], "point": n["point"]}
    ja = [n for n in na.values() if n["type"] == "junction"]
    jb = [n for n in nb.values() if n["type"] == "junction"]
    matched, used_b = [], set()
    for a in ja:
        hit = a2b.get(a["key"])
        if not hit or nb[hit[0]]["type"] != "junction":
            continue
        b = nb[hit[0]]
        used_b.add(b["key"])
        sim, info = base_node_similarity(a, b, cfg)
        if not info:
            _, info = multiset_junction_similarity(a["labels"] or a["rlabels"], b["labels"] or b["rlabels"],
                                                   cfg.junction_min_common)
        exact = bool(info.get("exact"))
        notes = []
        if info.get("unresolved"):
            notes.append(f"{info['unresolved']} unresolved label(s): extraction uncertainty, not a removal")
        if info.get("only_a") or info.get("only_b"):
            notes.append(f"labels only in A {info.get('only_a')}, only in B {info.get('only_b')}")
        if not info.get("related"):
            notes.append(f"matched through neighbour connectivity (support {hit[2]:.2f})")
        matched.append({"status": "exact" if exact else "probable", "similarity": hit[1], "neighbour_support": hit[2],
                        "junction_a": pub(a), "junction_b": pub(b), "common_labels": info.get("common"),
                        "notes": notes})
    only_a = [pub(a) for a in ja if a["key"] not in a2b or nb[a2b[a["key"]][0]]["type"] != "junction"]
    only_b = [pub(b) for b in jb if b["key"] not in used_b]
    return {"matched": matched, "only_in_a": only_a, "only_in_b": only_b,
            "summary": {"junctions_a": len(ja), "junctions_b": len(jb),
                        "exact": sum(1 for m in matched if m["status"] == "exact"),
                        "probable": sum(1 for m in matched if m["status"] == "probable"),
                        "only_in_a": len(only_a), "only_in_b": len(only_b)}}


# ===========================================================================
# Per-PDF processing
# ===========================================================================

def ep_json(ep: dict) -> dict:
    """Endpoint record; points in the displayed page frame."""
    out = {"type": ep["type"], "structural_id": final_id(ep), "original_structural_id": ep["original_structural_id"],
           "point": rpt(ep.get("point_display") or ep["point"]),
           "node_point": rpt(ep.get("node_point_display") or ep["node_point"])}
    if ep["type"] == "junction":
        out["depth1_labels"] = ep.get("labels")
        out["rebuilt_depth1_labels"] = ep.get("rebuilt_labels")
        out["rebuilt_structural_id"] = ep.get("rebuilt_structural_id")
    if ep["type"] in ("terminal", "element", "open"):
        out["name"] = ep.get("name")
        out["identity"] = ep.get("identity")
    return out


def seg_summary(s: dict, with_polyline: bool = False) -> dict:
    out = {"segment_index": s["index"], "segment_key": s["key"], "original_segment_key": s["orig_key"],
           "start": ep_json(s["start"]), "end": ep_json(s["end"]),
           "sequence": s["reb_sequence"], "original_sequence": s["orig_sequence"],
           "wire_numbers": s["reb_wires"], "original_wire_numbers": s["orig_wires"],
           "length": rnd(polyline_length(s["polyline"]))}
    if with_polyline:
        out["polyline"] = [rpt(p) for p in s.get("polyline_display") or s["polyline"]]
    return out


def junction_dot_diameter(segs: List[dict], circles: List[dict], text_unit: float) -> Optional[float]:
    """Median outer diameter of the junction dots (a filled circle, with any concentric rings)
    sitting on the junction endpoints of the segments."""
    idx = GridIndex(10.0)
    for i, c in enumerate(circles):
        idx.insert(i, c["center"][0], c["center"][1], c["center"][0], c["center"][1])
    r = 0.3 * text_unit
    found = {}
    for s in segs:
        for ep in (s["start"], s["end"]):
            if ep["type"] != "junction":
                continue
            p = ep["node_point"]
            key = (round(p[0], 1), round(p[1], 1))
            if key in found:
                continue
            here = [circles[i] for i in idx.query(p[0] - r, p[1] - r, p[0] + r, p[1] + r)
                    if dist(p, circles[i]["center"]) <= r and circles[i]["diameter"] <= 1.5 * text_unit]
            if any(c["filled"] for c in here):
                found[key] = max(c["diameter"] for c in here)
    ds = sorted(found.values())
    return ds[len(ds) // 2] if len(ds) >= 3 else None


def calibrate_pdf(segs: List[dict], texts: List[dict], vec: dict, frame: dict, Ut: float, cfg: V6Config) -> dict:
    """Profile of one PDF: units (text, wire pen, junction dot, layout) and label convention,
    measured on the segments of its JSON.  Explicit settings in cfg are kept."""
    tol = max(0.3, 0.1 * Ut)
    if cfg.wire_width is not None:
        Uw, src = cfg.wire_width, "explicit"
    else:
        Uw, src = wire_pen_from_segments(segs, vec["lines"], tol), "lines_under_segments"
        if Uw is None:
            Uw, src = REF_WIRE_UNIT * Ut / REF_TEXT_UNIT, "scaled_default"
    D = junction_dot_diameter(segs, vec["small_circles"], Ut)
    lab = calibrate_labels(segs, texts, Ut, cfg.axis_tol_deg)
    UL = layout_unit(Ut, Uw, D, lab["median_distance"])
    D = D if D else REF_DOT_UNIT * UL / REF_TEXT_UNIT
    side_h = cfg.side_h if cfg.side_h is not None else (lab["side_h"] if lab["side_h"] is not None else 1)
    side_v = cfg.side_v if cfg.side_v is not None else (lab["side_v"] if lab["side_v"] is not None else 1)
    med = lab["median_distance"]
    if cfg.label_max_side_override is not None:
        max_side = cfg.label_max_side_override
    elif med:
        max_side = max(2.4 * med, 1.2 * Ut)
    else:
        max_side = cfg.label_max_side * Ut / REF_TEXT_UNIT
    return {
        "deskew_deg": rnd(frame["theta"], 3), "axis_support": rnd(frame["support"], 3),
        "text_unit": rnd(Ut, 4), "wire_unit": Uw, "wire_width": Uw, "wire_width_source": src,
        "dot_unit": rnd(D, 4), "layout_unit": rnd(UL, 4),
        "label_median_distance": rnd(med, 4) if med else None, "label_max_side": rnd(max_side, 4),
        "inline_max_side": rnd(max(0.6 * Ut, 1.2 * (med or 0)), 4),
        "side_h": side_h, "side_v": side_v, "side_evidence": lab["evidence"],
        "text_axis_tol_deg": rnd(text_angle_tolerance(texts), 2),
    }


def process_pdf(tag: str, pdf_path: str, seg_path: str, page, cfg: V6Config) -> dict:
    """Load + validate one segment JSON against its PDF page, calibrate the PDF, then rebuild
    the text associations with thresholds in this PDF's own units."""
    data = load_segments_json(seg_path)
    segs = [normalize_segment(raw, i) for i, raw in enumerate(data["segments"])]
    if len({s["index"] for s in segs}) != len(segs):
        for i, s in enumerate(segs):
            s["index"] = i
    sha = sha256_file(pdf_path)
    drawings = page.get_drawings()
    frame = make_frame(page, cfg, drawings)
    probe = extract_texts(page, cfg, frame["M"], frame["view"])
    axis_tol = text_angle_tolerance(probe)
    all_texts = extract_texts(page, replace(cfg, text_axis_tol_deg=axis_tol), frame["M"], frame["view"])
    sizes = sorted(t["font_size"] for t in all_texts if t["is_numeric"] and t["font_size"] > 0) or \
        sorted(t["font_size"] for t in all_texts if t["font_size"] > 0)
    Ut = sizes[len(sizes) // 2] if sizes else REF_TEXT_UNIT     # first guess; refined below

    def extract(Ut: float):
        pre = replace(cfg, text_axis_tol_deg=axis_tol, text_unit=Ut, layout_unit=Ut, border_corner_tol=0.46 * Ut,
                      wire_unit=cfg.wire_width if cfg.wire_width is not None else REF_WIRE_UNIT * Ut / REF_TEXT_UNIT,
                      geometry_match_tol=max(cfg.geometry_match_tol * Ut / REF_TEXT_UNIT, 0.28 * Ut))
        border = detect_sheet_border(page, drawings, [tuple(t["center"]) for t in all_texts], pre, frame["M"])
        texts = [t for t in all_texts if not is_border_text(t, border)]   # zone markers never label wires
        for i, t in enumerate(texts):
            t["id"] = f"T{i:05d}"
        return pre, border, texts, extract_vectors(page, pre, drawings, border, frame["M"])

    pre, border, texts, vec = extract(Ut)
    ownership = validate_ownership(tag, data, segs, pdf_path, sha, page, vec, pre, frame)
    lu = label_text_unit(segs, texts)
    if lu is not None and abs(lu - Ut) > 0.1 * Ut:
        Ut = lu                           # e.g. zone numbers of several sheets outnumber the wire numbers
        pre, border, texts, vec = extract(Ut)
    prof = calibrate_pdf(segs, texts, vec, frame, Ut, cfg)
    pcfg = derive_pdf_config(cfg, prof)
    pcfg.projection_margin = cfg.projection_margin * Ut / REF_TEXT_UNIT
    flag_lines(vec, pcfg)
    segs_all = segs
    ignored = []
    for s in segs_all:                    # safety net for segment JSON that still contains the frame
        poly = s["polyline"]
        L = polyline_length(poly)
        on_border = sum(dist(a, b) for a, b in zip(poly, poly[1:]) if is_border_segment(a, b, border))
        if L > 0 and on_border >= cfg.border_segment_fraction * L:
            s["ignored"] = "sheet border / frame geometry, not a harness wire"
            ignored.append({"segment_index": s["index"], "segment_key": s["orig_key"], "reason": s["ignored"]})
    segs = [s for s in segs_all if not s.get("ignored")]
    groups = group_nodes(segs, pcfg.group_tol)
    # straight wire pieces (stations divide label spans)
    lidx, cidx = GridIndex(25.0), GridIndex(25.0)
    for i, l in enumerate(vec["lines"]):
        if l["significant"] and dist(l["p0"], l["p1"]) <= pcfg.station_line_max_length:
            lidx.insert(i, l["p0"][0], l["p0"][1], l["p1"][0], l["p1"][1])
    for i, c in enumerate(vec["small_circles"]):
        cidx.insert(i, c["center"][0], c["center"][1], c["center"][0], c["center"][1])
    pieces = []
    for s in segs:
        s["stations"] = detect_stations(s, vec, lidx, cidx, pcfg)
        s["pieces"] = build_pieces(tag, s, s["stations"], pcfg)
        pieces.extend(s["pieces"])
    assignments, text_diag = assign_numeric_labels(texts, pieces, pcfg)
    Rinv = frame["Rinv"]
    for a in assignments:                 # reported in the displayed page frame
        a["text_center"] = rpt(apply_matrix(Rinv, a["text_center"]))
    rebuild_sequences(segs, assignments, texts, pcfg.missing_position_window)
    term_diag = enrich_identities(segs, groups, texts, pcfg)
    rebuild_junction_ids(segs, groups)
    for s in segs:
        s["key"] = segment_key(final_id(s["start"]), final_id(s["end"]))
    tindex = GridIndex(20.0)
    for i, t in enumerate(texts):
        tindex.insert(i, *t["bbox"])

    def show(r):
        pts = [apply_matrix(Rinv, (r[0], r[1])), apply_matrix(Rinv, (r[2], r[1])),
               apply_matrix(Rinv, (r[2], r[3])), apply_matrix(Rinv, (r[0], r[3]))]
        return [rnd(min(p[0] for p in pts)), rnd(min(p[1] for p in pts)),
                rnd(max(p[0] for p in pts)), rnd(max(p[1] for p in pts))]
    border_report = {"frames": [show(f) for f in border["frames"]],
                     "drawing_area": show(border["inner"]) if border["inner"] else None,
                     "margin_texts_excluded": len(all_texts) - len(texts),
                     "ignored_segments": ignored}
    return {"tag": tag, "pdf": pdf_path, "sha256": sha, "segments_path": seg_path, "data": data, "segs": segs,
            "segs_all": segs_all, "border": border_report, "cfg": pcfg, "profile": prof, "frame": frame,
            "groups": groups, "texts": texts, "tindex": tindex, "pieces": pieces, "assignments": assignments,
            "text_diag": text_diag, "terminal_diag": term_diag, "ownership": ownership}


def corrected_json(res: dict, cfg: V6Config) -> dict:
    """Original JSON preserved; corrected / rebuilt fields added next to it."""
    data = copy.deepcopy(res["data"])
    Rinv = res["frame"]["Rinv"]
    for i, raw in enumerate(data["segments"]):
        s = res["segs_all"][i]                      # normalized in the same order as the JSON
        if s.get("ignored"):
            raw["ignored_reason"] = s["ignored"]
            continue
        conf_labels = [a["confidence"] for a in s["assignments"]]
        raw["segment_index"] = s["index"]
        raw["original_segment_key"] = s["orig_key"]
        raw["segment_key"] = s["key"]
        raw["original_sequence"] = s["orig_sequence"]
        raw["original_wire_numbers"] = s["orig_wires"]
        raw["rebuilt_sequence"] = s["reb_sequence"]
        raw["sequence"] = s["reb_sequence"]
        raw["wire_numbers"] = s["reb_wires"]
        raw["rebuilt_items"] = s["reb_items"]
        raw["numeric_text_assignments"] = s["assignments"]
        raw["wire_pieces"] = [{"piece_id": p["piece_id"], "piece_index": p["piece_index"],
                               "polyline_edge_index": p["polyline_edge_index"],
                               "start": rpt(apply_matrix(Rinv, p["start"])), "end": rpt(apply_matrix(Rinv, p["end"])),
                               "orientation": p["orientation"], "length": rnd(p["length"]),
                               "cum_start": rnd(p["cum_start"]), "cum_end": rnd(p["cum_end"])} for p in s["pieces"]]
        raw["stations_v6"] = [rnd(x) for x in s["stations"]]
        for side in ("start", "end"):
            ep = s[side]
            rec = raw.get(side) if isinstance(raw.get(side), dict) else {}
            rec = dict(rec)
            rec.update(ep_json(ep))
            raw[side] = rec
        raw["confidence"] = {
            "labels": rnd(sum(conf_labels) / len(conf_labels), 3) if conf_labels else None,
            "start_identity": s["start"].get("name_confidence"),
            "end_identity": s["end"].get("name_confidence"),
        }
    data["v6"] = {"script": os.path.basename(__file__), "version": VERSION, "config": res["cfg"].to_dict(),
                  "calibration": res["profile"],
                  "ownership_check": res["ownership"], "sheet_border": res["border"],
                  "text_assignment": res["text_diag"],
                  "terminal_assignment": res["terminal_diag"]}
    return data


# ===========================================================================
# Comparison driver
# ===========================================================================

def compare(A: dict, B: dict, cfg: V6Config) -> dict:
    sa_list, sb_list = A["segs"], B["segs"]
    na, nb = build_node_table(A), build_node_table(B)
    a2b, b2a = node_correspondence(na, nb, cfg)
    ctx = {"a2b": a2b, "b2a": b2a, "texts_a": A["texts"], "tindex_a": A["tindex"],
           "texts_b": B["texts"], "tindex_b": B["tindex"],
           "radius_a": A["cfg"].uncertain_text_radius, "radius_b": B["cfg"].uncertain_text_radius}
    cands = []
    for i, j in candidate_pairs(sa_list, sb_list, a2b):
        p = score_pair(sa_list[i], sb_list[j], cfg, ctx)
        if p is not None:
            cands.append((p["score"], i, j, p))
    cands.sort(key=lambda c: (-c[0], -c[3]["endpoint"]["mean"], -c[3]["content"], c[1], c[2]))
    matched: Dict[int, Tuple[int, dict]] = {}
    used_b = set()

    def greedy(min_score: float):
        for score, i, j, p in cands:
            if score < min_score or i in matched or j in used_b:
                continue
            matched[i] = (j, p)
            used_b.add(j)

    greedy(cfg.confident_pair_score)                                  # stage 1: confident pairs
    free_a = {i for i in range(len(sa_list)) if i not in matched}
    free_b = {j for j in range(len(sb_list)) if j not in used_b}
    merges = topology_recovery(sa_list, A["groups"], free_a, sb_list, free_b, cfg, x_is_a=True, ctx=ctx)
    for m in merges:
        free_a -= set(m["x"])
        free_b.discard(m["y"])
    splits = topology_recovery(sb_list, B["groups"], free_b, sa_list, free_a, cfg, x_is_a=False, ctx=ctx)
    topo_a, topo_b = set(), set()
    for m in merges:
        topo_a.update(m["x"])
        topo_b.add(m["y"])
    for m in splits:
        topo_b.update(m["x"])
        topo_a.add(m["y"])
    for score, i, j, p in cands:                                      # stage 3: remaining pairs
        if i in matched or j in used_b or i in topo_a or j in topo_b:
            continue
        matched[i] = (j, p)
        used_b.add(j)

    out = {"same": [], "probable_same": [], "changed": [], "review": [], "removed": [], "added": []}
    for i in sorted(matched):
        j, p = matched[i]
        sa, sb = sa_list[i], sb_list[j]
        res = classify_pair(sa, sb, p, ctx, cfg)
        cls = res.pop("classification")
        rec = {"segment_a": seg_summary(sa, cls not in ("same_exact", "same_probable")),
               "segment_b": seg_summary(sb, cls not in ("same_exact", "same_probable"))}
        rec.update(res)
        if cls == "same_exact":
            rec["classification"] = "same_exact"
            out["same"].append(rec)
        elif cls == "same_probable":
            rec["classification"] = "same_probable"
            out["probable_same"].append(rec)
        elif cls == "changed":
            rec["classification"] = "segment_content_changed"
            out["changed"].append(rec)
        else:
            rec["classification"] = "review"
            out["review"].append(rec)
    for m in merges + splits:
        segs_x = sa_list if m["x_is_a"] else sb_list
        segs_y = sb_list if m["x_is_a"] else sa_list
        xs = [seg_summary(segs_x[k], True) for k in m["x"]]
        ys = [seg_summary(segs_y[m["y"]], True)]
        out["review"].append({
            "classification": "review", "review_type": "route_merged" if m["x_is_a"] else "route_split",
            "segments_a": xs if m["x_is_a"] else ys, "segments_b": ys if m["x_is_a"] else xs,
            "confidence": rnd(m["score"], 4), "endpoint_confidence": rnd(m["endpoint"], 4),
            "content_similarity": rnd(m["content"], 4), "content_reversed": m["content_reversed"],
            "content_source": "rebuilt_sequences",
            "reason": (f"Route contents are present in both PDFs, but node {m['node_id']} divides the route "
                       f"only in PDF {'A' if m['x_is_a'] else 'B'} (junction removed or added).")})
    for i, sa in enumerate(sa_list):
        if i not in matched and i not in topo_a:
            out["removed"].append({"classification": "entire_branch_removed", "segment_a": seg_summary(sa, True),
                                   "reason": "No sufficiently compatible segment exists in PDF B after fuzzy "
                                             "endpoint, UNLABELLED, OPEN, forward/reverse content, original-"
                                             "sequence and route merge/split matching."})
    for j, sb in enumerate(sb_list):
        if j not in used_b and j not in topo_b:
            out["added"].append({"classification": "entire_branch_added", "segment_b": seg_summary(sb, True),
                                 "reason": "No sufficiently compatible segment exists in PDF A after fuzzy "
                                           "endpoint, UNLABELLED, OPEN, forward/reverse content, original-"
                                           "sequence and route merge/split matching."})
    prefixes = {"same": "SAME", "probable_same": "PROB", "changed": "CHG", "review": "REV",
                "removed": "DEL", "added": "ADD"}
    for cat, pre in prefixes.items():
        out[cat] = [{"finding_id": f"{pre}-{k:03d}", **rec} for k, rec in enumerate(out[cat], 1)]
    out["junctions"] = junction_report(na, nb, a2b, cfg)
    return out


# ===========================================================================
# Marked images
# ===========================================================================

COLORS = {  # BGR
    "changed": (0, 140, 255),      # orange
    "review": (0, 215, 255),       # yellow
    "removed": (0, 0, 230),        # red
    "added": (0, 170, 0),          # green
    "same": (200, 160, 60),        # optional, light blue
}


def zp(p, zoom: float) -> Tuple[int, int]:
    return (int(round(p[0] * zoom)), int(round(p[1] * zoom)))


def put_tag(img, text: str, org: Tuple[int, int], color, scale: float = 0.5) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(org[0], img.shape[1] - w - 4))
    y = max(h + 4, min(org[1], img.shape[0] - 4))
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base + 1), (255, 255, 255), -1)
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base + 1), color, 1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_segment(img, poly, color, zoom: float, tag: Optional[str] = None, thin: bool = False) -> None:
    """Complete logical polyline (bends included), white outline, endpoint circles."""
    pts = np.array([zp(p, zoom) for p in poly], dtype=np.int32)
    if len(pts) < 2:
        return
    t = max(2, int(round((0.8 if thin else 1.6) * zoom)))
    cv2.polylines(img, [pts], False, (255, 255, 255), t + 4, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, color, t, cv2.LINE_AA)
    r = t + 3
    for p in (pts[0], pts[-1]):
        c = (int(p[0]), int(p[1]))
        cv2.circle(img, c, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
    if tag:
        L = polyline_length(poly)
        half, cum = L / 2.0, 0.0
        mid = poly[0]
        for a, b in zip(poly, poly[1:]):
            l = dist(a, b)
            if cum + l >= half and l > 0:
                f = (half - cum) / l
                mid = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
                break
            cum += l
        m = zp(mid, zoom)
        put_tag(img, tag, (m[0] + 6, m[1] - 8), color, 0.45 if zoom < 3 else 0.55)


def render_marked(page, zoom: float, marks: List[Tuple[List[Tuple[float, float]], tuple, str, bool]],
                  title: str, path: str) -> None:
    """Draw the marks on the page and put the legend in a strip BELOW the drawing.

    The strip never hides drawing content (important for sheets without a border, where
    the harness reaches the page corners); drawing pixels keep the mapping point * zoom.
    """
    img = render_page_bgr(page, zoom)
    for poly, color, tag, thin in sorted(marks, key=lambda m: m[3], reverse=True):
        draw_segment(img, poly, color, zoom, tag, thin)
    legend = [("Orange: confirmed segment-content change", COLORS["changed"]),
              ("Yellow: annotation / extraction review", COLORS["review"]),
              ("Red: entire branch removed from PDF A", COLORS["removed"]),
              ("Green: entire branch added in PDF B", COLORS["added"])]
    h = 46 + 26 * len(legend)
    strip = np.full((h, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.line(strip, (0, 1), (img.shape[1] - 1, 1), (120, 120, 120), 1)
    cv2.putText(strip, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    for k, (txt, col) in enumerate(legend):
        y = 58 + 26 * k
        cv2.line(strip, (12, y - 6), (46, y - 6), col, 5)
        cv2.putText(strip, txt, (58, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    if not cv2.imwrite(path, np.vstack([img, strip])):
        raise RuntimeError(f"could not write image {path}")


def build_marks(cmp_res: dict, A: dict, B: dict, cfg: V6Config):
    poly_a = {s["index"]: s.get("polyline_display") or s["polyline"] for s in A["segs"]}
    poly_b = {s["index"]: s.get("polyline_display") or s["polyline"] for s in B["segs"]}
    ma, mb = [], []
    for rec in cmp_res["changed"]:
        ma.append((poly_a[rec["segment_a"]["segment_index"]], COLORS["changed"], rec["finding_id"], False))
        mb.append((poly_b[rec["segment_b"]["segment_index"]], COLORS["changed"], rec["finding_id"], False))
    for rec in cmp_res["review"]:
        sa = rec.get("segments_a") or [rec["segment_a"]]
        sb = rec.get("segments_b") or [rec["segment_b"]]
        for s in sa:
            ma.append((poly_a[s["segment_index"]], COLORS["review"], rec["finding_id"], False))
        for s in sb:
            mb.append((poly_b[s["segment_index"]], COLORS["review"], rec["finding_id"], False))
    for rec in cmp_res["removed"]:
        ma.append((poly_a[rec["segment_a"]["segment_index"]], COLORS["removed"], rec["finding_id"], False))
    for rec in cmp_res["added"]:
        mb.append((poly_b[rec["segment_b"]["segment_index"]], COLORS["added"], rec["finding_id"], False))
    if cfg.mark_same:
        for cat in ("same", "probable_same"):
            for rec in cmp_res[cat]:
                ma.append((poly_a[rec["segment_a"]["segment_index"]], COLORS["same"], None, True))
                mb.append((poly_b[rec["segment_b"]["segment_index"]], COLORS["same"], None, True))
    return ma, mb


# ===========================================================================
# Self checks
# ===========================================================================

def self_check(A: dict, B: dict) -> List[str]:
    """Verify global-ownership invariants before writing results."""
    problems = []
    for R in (A, B):
        ids = [a["text_id"] for a in R["assignments"]]
        if len(ids) != len(set(ids)):
            problems.append(f"PDF {R['tag']}: a numeric text ID is assigned more than once")
        per_node = {}
        for s in R["segs"]:
            for ep in (s["start"], s["end"]):
                tid = (ep.get("identity") or {}).get("text_id")
                if tid:
                    per_node.setdefault(tid, set()).add(ep["node_key"])
        if any(len(v) > 1 for v in per_node.values()):
            problems.append(f"PDF {R['tag']}: a terminal text ID identifies more than one endpoint")
        for s in R["segs"]:
            pts = [p for pc in s["pieces"] for p in (pc["start"], pc["end"])]
            if pts and (dist(pts[0], s["polyline"][0]) > 1e-6 or dist(pts[-1], s["polyline"][-1]) > 1e-6):
                problems.append(f"PDF {R['tag']}: polyline order not preserved in segment {s['index']}")
    return problems


# ===========================================================================
# Command line
# ===========================================================================

SIDE_CHOICES_H = {"auto": None, "below": 1, "above": -1, "on-wire": 0}
SIDE_CHOICES_V = {"auto": None, "right": 1, "left": -1, "on-wire": 0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Script 3 (v6): strict global text ownership and structural comparison of two harness PDFs.")
    ap.add_argument("--pdf-a", required=True, help="Baseline / old PDF.")
    ap.add_argument("--pdf-b", required=True, help="New / comparison PDF.")
    ap.add_argument("--segments-a", required=True, help="Segment JSON generated from --pdf-a.")
    ap.add_argument("--segments-b", required=True, help="Segment JSON generated from --pdf-b.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--page", type=int, default=0, help="Zero-based page number (default 0).")
    ap.add_argument("--page-a", type=int, help="Zero-based page of PDF A (default: --page).")
    ap.add_argument("--page-b", type=int, help="Zero-based page of PDF B (default: --page).")
    ap.add_argument("--zoom", type=float, default=3.0, help="Rendering zoom (default 3.0).")
    ap.add_argument("--wire-width", default="auto",
                    help="Harness wire stroke width in pt, or 'auto' (default): the pen of the PDF lines under "
                         "the segment polylines, measured per PDF.")
    ap.add_argument("--label-max-side", default="auto",
                    help="Max distance (pt) from wire to numeric label centre on the required side, or 'auto' "
                         "(default): 2.4 x the measured typical label distance of each PDF.")
    ap.add_argument("--label-side-h", choices=SIDE_CHOICES_H, default="auto",
                    help="Side of horizontal wires that carries their numbers (default auto = measured, "
                         "falling back to below).")
    ap.add_argument("--label-side-v", choices=SIDE_CHOICES_V, default="auto",
                    help="Side of vertical wires that carries their numbers (default auto = measured, "
                         "falling back to right).")
    ap.add_argument("--no-deskew", action="store_true",
                    help="Do not straighten a drawing that is rotated off the page axes.")
    ap.add_argument("--projection-margin", type=float, default=2.0,
                    help="Max distance a label centre may project beyond a straight piece end, in pt of the "
                         "reference drawing (rescaled to each PDF's text size).")
    ap.add_argument("--mark-same", action="store_true", help="Also draw same / probable-same segments (thin blue).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = V6Config()
    ww = str(args.wire_width).strip().lower()
    lm = str(args.label_max_side).strip().lower()
    try:
        cfg.wire_width = None if ww in ("none", "auto", "") else float(ww)
        cfg.label_max_side_override = None if lm in ("auto", "") else float(lm)
    except ValueError:
        print("ERROR: --wire-width and --label-max-side take a number or 'auto' "
              f"(got {args.wire_width!r}, {args.label_max_side!r})", file=sys.stderr)
        return 2
    if (cfg.wire_width is not None and cfg.wire_width <= 0) or \
            (cfg.label_max_side_override is not None and cfg.label_max_side_override <= 0):
        print("ERROR: --wire-width and --label-max-side must be positive", file=sys.stderr)
        return 2
    cfg.wire_width_auto = cfg.wire_width is None
    cfg.side_h, cfg.side_v = SIDE_CHOICES_H[args.label_side_h], SIDE_CHOICES_V[args.label_side_v]
    cfg.deskew = not args.no_deskew
    cfg.projection_margin = float(args.projection_margin)
    cfg.mark_same = bool(args.mark_same)

    for flag, path in (("--pdf-a", args.pdf_a), ("--pdf-b", args.pdf_b),
                       ("--segments-a", args.segments_a), ("--segments-b", args.segments_b)):
        if not os.path.isfile(path):
            print(f"ERROR: {flag} file not found: {path}", file=sys.stderr)
            return 2
    pages = {"A": args.page if args.page_a is None else args.page_a,
             "B": args.page if args.page_b is None else args.page_b}
    if min(pages.values()) < 0:
        print("ERROR: page numbers must be zero or positive", file=sys.stderr)
        return 2
    if args.zoom <= 0:
        print("ERROR: --zoom must be positive", file=sys.stderr)
        return 2
    if os.path.abspath(args.segments_a) == os.path.abspath(args.segments_b) and \
            os.path.abspath(args.pdf_a) != os.path.abspath(args.pdf_b):
        print("ERROR: the same segment JSON was given for two different PDFs", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    doc_a = doc_b = None
    try:
        doc_a = fitz.open(args.pdf_a)
        doc_b = fitz.open(args.pdf_b)
        for tag, doc, path in (("A", doc_a, args.pdf_a), ("B", doc_b, args.pdf_b)):
            if pages[tag] >= doc.page_count:
                raise ValueError(f"page {pages[tag]} out of range for PDF {tag} '{path}' "
                                 f"({doc.page_count} page(s), zero-based)")
        page_a, page_b = doc_a[pages["A"]], doc_b[pages["B"]]

        A = process_pdf("A", args.pdf_a, args.segments_a, page_a, cfg)
        B = process_pdf("B", args.pdf_b, args.segments_b, page_b, cfg)
        problems = self_check(A, B)
        if problems:
            raise RuntimeError("self check failed: " + "; ".join(problems))
        cmp_res = compare(A, B, cfg)

        out_a = os.path.join(args.out, "pdf_a_segments_corrected_v6.json")
        out_b = os.path.join(args.out, "pdf_b_segments_corrected_v6.json")
        out_cmp = os.path.join(args.out, "segment_comparison_v6.json")
        img_a = os.path.join(args.out, "pdf_a_segment_changes_v6.png")
        img_b = os.path.join(args.out, "pdf_b_segment_changes_v6.png")
        with open(out_a, "w", encoding="utf-8") as fh:
            json.dump(corrected_json(A, cfg), fh, indent=2)
        with open(out_b, "w", encoding="utf-8") as fh:
            json.dump(corrected_json(B, cfg), fh, indent=2)

        review_types = Counter(r.get("review_type") for r in cmp_res["review"])
        summary = {
            "segments_a": len(A["segs"]),
            "segments_b": len(B["segs"]),
            "same_exact": len(cmp_res["same"]),
            "same_probable": len(cmp_res["probable_same"]),
            "changed": len(cmp_res["changed"]),
            "review": len(cmp_res["review"]),
            "entire_branches_removed": len(cmp_res["removed"]),
            "entire_branches_added": len(cmp_res["added"]),
            "review_types": dict(review_types),
            "junctions": cmp_res["junctions"]["summary"],
            "numeric_labels_assigned_a": A["text_diag"]["assigned"],
            "numeric_labels_assigned_b": B["text_diag"]["assigned"],
            "sheet_frames_a": len(A["border"]["frames"]),
            "sheet_frames_b": len(B["border"]["frames"]),
            "ignored_border_segments_a": len(A["border"]["ignored_segments"]),
            "ignored_border_segments_b": len(B["border"]["ignored_segments"]),
        }
        result = {
            "summary": summary,
            "same": cmp_res["same"],
            "probable_same": cmp_res["probable_same"],
            "changed": cmp_res["changed"],
            "review": cmp_res["review"],
            "removed": cmp_res["removed"],
            "added": cmp_res["added"],
            "junctions": cmp_res["junctions"],
            "inputs": {
                "pdf_a": os.path.abspath(args.pdf_a), "pdf_b": os.path.abspath(args.pdf_b),
                "pdf_a_sha256": A["sha256"], "pdf_b_sha256": B["sha256"],
                "segments_a": os.path.abspath(args.segments_a), "segments_b": os.path.abspath(args.segments_b),
                "page": args.page, "page_a": pages["A"], "page_b": pages["B"], "zoom": args.zoom,
                "ownership_check_a": A["ownership"], "ownership_check_b": B["ownership"],
                "sheet_border_a": A["border"], "sheet_border_b": B["border"],
                "calibration_a": A["profile"], "calibration_b": B["profile"],
            },
            "diagnostics": {
                "text_assignment_a": A["text_diag"], "text_assignment_b": B["text_diag"],
                "terminal_assignment_a": A["terminal_diag"], "terminal_assignment_b": B["terminal_diag"],
                "numeric_text_assignments_a": A["assignments"], "numeric_text_assignments_b": B["assignments"],
            },
            "config": cfg.to_dict(),
            "config_a": A["cfg"].to_dict(), "config_b": B["cfg"].to_dict(),
            "generator": {"script": os.path.basename(__file__), "version": VERSION},
        }
        with open(out_cmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        marks_a, marks_b = build_marks(cmp_res, A, B, cfg)
        render_marked(page_a, args.zoom, marks_a, "PDF A (baseline) - structural segment comparison v6", img_a)
        render_marked(page_b, args.zoom, marks_b, "PDF B (new) - structural segment comparison v6", img_b)
    except (OwnershipError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if doc_a is not None:
            doc_a.close()
        if doc_b is not None:
            doc_b.close()

    print("Structural harness comparison v6")
    print(f"  PDF A          : {args.pdf_a}  ({summary['segments_a']} segments, "
          f"{summary['numeric_labels_assigned_a']} numeric labels owned, check: {A['ownership']['method']})")
    print(f"  PDF B          : {args.pdf_b}  ({summary['segments_b']} segments, "
          f"{summary['numeric_labels_assigned_b']} numeric labels owned, check: {B['ownership']['method']})")
    for tag, R in (("A", A), ("B", B)):
        pr = R["profile"]
        print(f"  Calibration {tag}  : text {pr['text_unit']}  wire pen {pr['wire_unit']} ({pr['wire_width_source']})  "
              f"dot {pr['dot_unit']}  skew {pr['deskew_deg']} deg  labels h={pr['side_h']} v={pr['side_v']} "
              f"max {pr['label_max_side']}")
    print(f"  Sheet border   : A {summary['sheet_frames_a']} frame(s), B {summary['sheet_frames_b']} frame(s); "
          f"margin texts excluded A {A['border']['margin_texts_excluded']}, B {B['border']['margin_texts_excluded']}; "
          f"border segments ignored A {summary['ignored_border_segments_a']}, B {summary['ignored_border_segments_b']}")
    print(f"  Same exact     : {summary['same_exact']}")
    print(f"  Same probable  : {summary['same_probable']}")
    print(f"  Changed        : {summary['changed']}")
    print(f"  Review         : {summary['review']}  {summary['review_types'] or ''}")
    print(f"  Removed (A)    : {summary['entire_branches_removed']}")
    print(f"  Added (B)      : {summary['entire_branches_added']}")
    j = summary["junctions"]
    print(f"  Junctions      : exact {j['exact']}, probable {j['probable']}, only A {j['only_in_a']}, only B {j['only_in_b']}")
    for rec in cmp_res["changed"]:
        print(f"    {rec['finding_id']}: {rec['segment_a']['segment_key']}  ->  {'; '.join(rec['changes'])}")
    for rec in cmp_res["review"]:
        print(f"    {rec['finding_id']}: review ({rec.get('review_type')})")
    for rec in cmp_res["removed"]:
        print(f"    {rec['finding_id']}: removed {rec['segment_a']['segment_key']} {rec['segment_a']['sequence']}")
    for rec in cmp_res["added"]:
        print(f"    {rec['finding_id']}: added   {rec['segment_b']['segment_key']} {rec['segment_b']['sequence']}")
    for p in (out_a, out_b, out_cmp, img_a, img_b):
        print(f"  Wrote          : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
