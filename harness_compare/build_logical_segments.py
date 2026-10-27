#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2 - Connectivity graph and logical (unbranched) harness segments.

Pipeline
--------
    Script 1  junction_depth1_ids.py              (extraction engine, imported here)
    Script 2  build_logical_segments.py           (this file)
    Script 3  compare_and_render_segments_v6.py   (strict text ownership + final comparison)

What this script does
---------------------
Script 1's extraction is applied independently to PDF A and PDF B.  For each PDF:

1.  Wire endpoints within a small tolerance are merged into graph vertices;
    the original wire vectors are the graph edges.
2.  Structural boundary nodes are created for depth-1 junctions (DOT1[...]),
    large hollow / solid circular elements, degree-1 terminal endpoints,
    open endpoints and unexpected branch points.  Terminal and element nodes
    initially use meaningful nearby non-numeric text when available.
3.  From every boundary node every unvisited outgoing wire edge is followed
    through ordinary degree-2 points (geometric bends only) until another
    boundary node is reached.  Each boundary-to-boundary path is one logical
    segment.  The complete polyline, the graph points and every original wire
    line ID are stored.  Bends never become nodes, elements or boundaries.
4.  A direction-independent key is built from the sorted endpoint IDs, e.g.
    SEG[DOT1[125|208|430]<->TERMINAL[X101]].
5.  Numeric labels and generic annotations (CLAMP, ROUTE, LABEL, MARKER, ...)
    and inline components are assigned to segments and ordered along the
    complete route, e.g. WIRE:125, ELEMENT:CLAMP, WIRE:208.
6.  An initial key-based comparison (same / changed / removed / added, with
    direct and reversed content order) is written together with marked images.

The segment JSON files written here are the inputs of Script 3.  They record a
SHA-256 fingerprint of their PDF so that Script 3 can refuse a JSON that was
generated from a different drawing.

Example
-------
    python build_logical_segments.py --pdf-a old.pdf --pdf-b new.pdf --out segments_pair
    -> segments_pair/pdf_a_segments.json, segments_pair/pdf_b_segments.json, ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import junction_depth1_ids as eng  # noqa: E402  (Script 1, the shared extraction engine)

SCHEMA = "harness-segments/1"


@dataclass
class SegmentConfig:
    """Distances in pt for the reference drawing; scaled() rescales them to a PDF's own units."""
    station_tol: float = 0.8              # stations (breakpoints, leader feet, ticks) on the wire
    station_line_max_length: float = 40.0
    leader_wire_tol: float = 1.0          # leader foot must touch the wire
    leader_text_reach: float = 12.0       # leader head must be this close to the annotation text
    annotation_fallback_dist: float = 6.0
    node_annotation_tol: float = 1.0
    component_min_size: float = 3.0
    component_max_size: float = 60.0
    component_text_radius: float = 15.0
    initial_label_max_dist: float = 10.5  # Script 2 initial (distance based) association
    end_component_reach: float = 40.0     # named box this close to an unnamed wire end names it
    caption_min: float = 2.0              # caption above an annotation keyword, distance window
    caption_max: float = 14.0
    station_merge: float = 2.0
    end_margin: float = 1.0               # stations / annotations this close to a route end belong to the node
    name_below_penalty: float = 4.0
    name_below_slack: float = 3.0
    name_align_weight: float = 0.25       # per pt off the symbol's line in the text's reading frame
    name_align_dead: float = 2.7          # ... beyond half a text height
    weak_name_radius: float = 12.0
    collinear_deg: float = 0.5            # polyline vertices bending less than this are dropped

    @classmethod
    def scaled(cls, cfg: "eng.EngineConfig") -> "SegmentConfig":
        """Reference distances x (layout unit / reference), wire-pen and text distances by their units."""
        g = cfg.layout_unit / eng.REF_TEXT_UNIT
        t = cfg.text_unit / eng.REF_TEXT_UNIT
        return cls(
            station_tol=max(1.5 * cfg.wire_unit, 0.16 * cfg.layout_unit),
            station_line_max_length=40.0 * g,
            leader_wire_tol=0.5 * cfg.wire_unit + 0.15 * cfg.layout_unit,   # foot on the wire edge
            leader_text_reach=12.0 * g,
            annotation_fallback_dist=6.0 * g,
            node_annotation_tol=1.0 * g,
            component_min_size=3.0 * g,
            component_max_size=60.0 * g,
            component_text_radius=15.0 * g,
            initial_label_max_dist=cfg.label_max_side,
            end_component_reach=40.0 * g,
            caption_min=2.0 * t,
            caption_max=14.0 * t,
            station_merge=2.0 * g,
            end_margin=1.0 * g,
            name_below_penalty=4.0 * g,
            name_below_slack=3.0 * g,
            name_align_dead=0.5 * cfg.text_unit,
            weak_name_radius=12.0 * g,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------

def polyline_length(poly: Sequence[Sequence[float]]) -> float:
    return sum(eng.dist(a, b) for a, b in zip(poly, poly[1:]))


def project_on_polyline(p: Sequence[float], poly: Sequence[Sequence[float]]) -> Tuple[float, float, int, Tuple[float, float]]:
    """(distance, arc length s, piece index, foot point) of p projected on a polyline."""
    best = (float("inf"), 0.0, 0, (poly[0][0], poly[0][1]))
    cum = 0.0
    for i, (a, b) in enumerate(zip(poly, poly[1:])):
        d, t = eng.point_segment_distance(p, a, b)
        L = eng.dist(a, b)
        if d < best[0]:
            foot = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            best = (d, cum + t * L, i, foot)
        cum += L
    return best


class SegmentIndex:
    """Grid index over the straight pieces of all segment polylines."""

    def __init__(self, segments: List[dict], cell: float = 25.0):
        self.segments = segments
        self.grid = eng.GridIndex(cell)
        self.pieces: List[Tuple[int, Tuple[float, float], Tuple[float, float], float]] = []
        for pos, seg in enumerate(segments):
            cum = 0.0
            poly = seg["polyline"]
            for a, b in zip(poly, poly[1:]):
                self.pieces.append((pos, a, b, cum))
                self.grid.insert(len(self.pieces) - 1, a[0], a[1], b[0], b[1])
                cum += eng.dist(a, b)

    def nearest(self, p: Sequence[float], maxd: float):
        """(distance, segment, arc position, foot point) of the closest route within maxd."""
        best = None
        for k in self.grid.query(p[0] - maxd, p[1] - maxd, p[0] + maxd, p[1] + maxd):
            pos, a, b, cum = self.pieces[k]
            d, t = eng.point_segment_distance(p, a, b)
            if d <= maxd and (best is None or d < best[0]):
                foot = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                best = (d, self.segments[pos], cum + t * eng.dist(a, b), foot)
        return best

    def segments_near_box(self, x0: float, y0: float, x1: float, y1: float) -> List[int]:
        return sorted({self.pieces[k][0] for k in self.grid.query(x0, y0, x1, y1)})


def clean_polyline(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not out or eng.dist(out[-1], p) > 1e-3:
            out.append((float(p[0]), float(p[1])))
    return out


# ---------------------------------------------------------------------------
# Node naming (initial, per endpoint, no global ownership - Script 3 refines)
# ---------------------------------------------------------------------------

def canonical_name(name: str) -> str:
    """'6711VN / B2' -> '6711VN/B2' (stable identity text)."""
    return re.sub(r"\s*/\s*", "/", " ".join(str(name).split())).upper()


def name_candidates(point: Sequence[float], texts: List[dict], tindex: eng.GridIndex, radius: float,
                    allow_spec: bool = False, scfg: Optional["SegmentConfig"] = None,
                    inward: Optional[Tuple[float, float]] = None) -> List[Tuple[float, dict]]:
    """Meaningful identifiers near a point, best first.

    Penalised: text below the point in the text's own reading frame (names are written
    above / level with their symbol; this stays true when the whole drawing is rotated)
    and text lying over the wire side of a wire end (inward).
    """
    scfg = scfg or SegmentConfig()
    out = []
    for ti in tindex.query(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius):
        t = texts[ti]
        d = eng.rect_distance(point, t["bbox"])
        if d > radius:
            continue
        norm = t["norm"]
        ok = eng.is_meaningful_identifier(norm) or (allow_spec and eng.is_spec_code(norm))
        if not ok:
            continue
        up = (t["dir"][1], -t["dir"][0])
        v = (t["center"][0] - point[0], t["center"][1] - point[1])
        score = d
        below = -(v[0] * up[0] + v[1] * up[1])
        if below > 0:          # ramp, not a step: no cliff for names right at the slack
            score += scfg.name_below_penalty * min(1.0, below / (2.0 * scfg.name_below_slack))
        # names are printed level with their symbol: small cost for being off that line
        score += scfg.name_align_weight * max(0.0, abs(below) - scfg.name_align_dead)
        if inward is not None and v[0] * inward[0] + v[1] * inward[1] > scfg.name_below_slack:
            score += 2.0 * scfg.name_below_penalty
        out.append((score, t))
    out.sort(key=lambda x: (x[0], x[1]["id"]))
    return out


def weak_name(point: Sequence[float], texts: List[dict], tindex: eng.GridIndex, radius: float) -> Optional[dict]:
    """Fallback name such as BRAID for a box terminal without an alphanumeric identifier."""
    best = None
    for ti in tindex.query(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius):
        t = texts[ti]
        n = t["norm"]
        if t["is_numeric"] or eng.is_generic_size(n) or eng.is_spec_code(n) or len(n) < 2:
            continue
        kw = eng.annotation_keyword(n)
        if kw and kw != "BRAID":
            continue
        d = eng.rect_distance(point, t["bbox"])
        if d <= radius and (best is None or d < best[0]):
            best = (d, t)
    return best[1] if best else None


def _inward(graph: dict, n: dict) -> Optional[Tuple[float, float]]:
    """Unit vector from a wire end into its wire (None for nodes with several wires)."""
    if len(n["edges"]) != 1:
        return None
    _, near, far = eng.edge_far(graph, n["edges"][0], n["id"])
    return eng.unit((far[0] - near[0], far[1] - near[1]))


def assign_node_identities(ext: dict, cfg: eng.EngineConfig, scfg: "SegmentConfig") -> None:
    """Structural IDs for boundary nodes (junction IDs come from Script 1's depth-1 tracing).

    Names are chosen with global ownership: all node / text candidates are ranked and one
    text object names at most one node, so a neighbour's name is never reused.
    """
    texts = ext["texts"]
    tindex = eng.build_text_index(texts, numeric_only=False)
    graph = ext["graph"]
    nodes = graph["nodes"]
    cands = []
    for n in nodes:
        n.setdefault("structural_id", None)
        n["name"] = None
        n["name_text_id"] = None
        n["name_confidence"] = 0.0
        if n["kind"] in ("terminal", "open"):
            r = cfg.terminal_text_radius
            for sc, t in name_candidates(n["point"], texts, tindex, r, scfg=scfg, inward=_inward(graph, n)):
                cands.append((sc, n["id"], t["id"], t, r))
        elif n["kind"] == "element":
            r = cfg.element_text_radius + (n["symbol"]["outer_diameter"] / 2.0 if n.get("symbol") else 0.0)
            for sc, t in name_candidates(n["point"], texts, tindex, r, scfg=scfg):
                cands.append((sc, n["id"], t["id"], t, r))
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    used_t, named = set(), set()
    for sc, nid, tid, t, r in cands:
        if nid in named or tid in used_t:
            continue
        named.add(nid)
        used_t.add(tid)
        n = nodes[nid]
        n["name"], n["name_text_id"] = t["norm"], tid
        n["name_confidence"] = eng.rnd(max(0.2, 1.0 - sc / (2 * r)), 3)
        if n["kind"] == "open":
            n["kind"] = "terminal"              # an identifier next to a wire end names a terminal
    for n in nodes:
        if n["kind"] in ("terminal", "open") and n["name"] is None:
            t = weak_name(n["point"], texts, tindex, scfg.weak_name_radius)
            if t is not None and t["id"] not in used_t:
                used_t.add(t["id"])
                n["name"], n["name_text_id"], n["name_confidence"] = t["norm"], t["id"], 0.3
                n["kind"] = "terminal"
    for n in nodes:
        kind = n["kind"]
        if n["name"]:
            n["name"] = canonical_name(n["name"])
        if kind in ("junction", "branch"):
            continue  # structural_id already set by compute_depth1_ids
        if kind == "terminal":
            n["structural_id"] = f"TERMINAL[{n['name'] or 'UNIDENTIFIED'}]"
        elif kind == "element":
            n["structural_id"] = f"ELEMENT[{n['name'] or 'UNNAMED'}]"
        elif kind == "open":
            n["structural_id"] = f"OPEN[{n['id']}]"     # cluster index - NOT a stable identity


# ---------------------------------------------------------------------------
# Segment tracing
# ---------------------------------------------------------------------------

def simplify_polyline(pts: List[Tuple[float, float]], collinear_deg: float) -> List[Tuple[float, float]]:
    """Drop vertices where the route does not bend (segmentation artefacts, pass points on a
    straight run).  Real bends stay; the order of points is preserved."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        u, v = eng.unit((b[0] - a[0], b[1] - a[1])), eng.unit((c[0] - b[0], c[1] - b[1]))
        if u[0] * v[0] + u[1] * v[1] >= math.cos(math.radians(collinear_deg)):
            continue
        out.append(b)
    out.append(pts[-1])
    return out


def trace_segments(graph: dict, collinear_deg: float = 0.5) -> List[dict]:
    """Follow every boundary-to-boundary route through degree-2 points."""
    nodes, edges = graph["nodes"], graph["edges"]
    boundary = {n["id"] for n in nodes if n["kind"] in eng.BOUNDARY_KINDS}
    visited = set()
    segments: List[dict] = []

    def walk(start: int, first_edge: int, loop: bool = False) -> dict:
        cur, eid = start, first_edge
        pts: List[Tuple[float, float]] = []
        gpts = [start]
        eids: List[int] = []
        wire_ids: List[str] = []
        while True:
            visited.add(eid)
            far, near_pt, far_pt = eng.edge_far(graph, eid, cur)
            if not pts:
                pts.append(near_pt)
            eids.append(eid)
            for w in edges[eid]["wire_ids"]:
                if w not in wire_ids:
                    wire_ids.append(w)
            gpts.append(far)
            if far in boundary or (loop and far == start):
                pts.append(far_pt)
                break
            pts.append(nodes[far]["point"])      # a bend / pass-through point
            nxt = [e for e in nodes[far]["edges"] if e != eid and e not in visited]
            if not nxt:
                pts[-1] = far_pt
                break
            cur, eid = far, nxt[0]
        return {"start_node": start, "end_node": gpts[-1], "polyline": simplify_polyline(clean_polyline(pts), collinear_deg),
                "graph_points": gpts, "edge_ids": eids, "vector_line_ids": wire_ids}

    for nid in sorted(boundary):
        for eid in nodes[nid]["edges"]:
            if eid not in visited:
                segments.append(walk(nid, eid))
    for e in edges:                              # closed loops without any boundary node
        if e["id"] not in visited:
            segments.append(walk(e["u"], e["id"], loop=True))
    for i, s in enumerate(segments):
        s["segment_index"] = i
    return segments


# ---------------------------------------------------------------------------
# Stations, annotations, inline components
# ---------------------------------------------------------------------------

def build_station_index(ext: dict, scfg: SegmentConfig) -> dict:
    """Grids of the objects that can divide a straight run: small hollow circles and short lines."""
    circ, lines = eng.GridIndex(25.0), eng.GridIndex(25.0)
    circles = [s for s in ext["symbols"] if s.get("kind") in ("hollow_symbol", "two_direction_dot")]
    for i, c in enumerate(circles):
        circ.insert(i, c["center"][0], c["center"][1], c["center"][0], c["center"][1])
    cand = [o for o in ext["vectors"]["other_lines"]
            if eng.is_significant_line(o, ext["cfg"]) and eng.dist(o["p0"], o["p1"]) <= scfg.station_line_max_length]
    for i, o in enumerate(cand):
        lines.insert(i, o["p0"][0], o["p0"][1], o["p1"][0], o["p1"][1])
    return {"circles": circles, "circle_grid": circ, "lines": cand, "line_grid": lines}


def detect_stations(poly: List[Tuple[float, float]], sindex: dict, scfg: SegmentConfig) -> List[dict]:
    """Points along the route that divide label spans: breakpoints, leader feet, ticks."""
    L = polyline_length(poly)
    if L <= 0:
        return []
    out = []
    tol = scfg.station_tol + 0.5
    seen_c, seen_l = set(), set()
    for p, q in zip(poly, poly[1:]):
        x0, y0, x1, y1 = min(p[0], q[0]) - tol, min(p[1], q[1]) - tol, max(p[0], q[0]) + tol, max(p[1], q[1]) + tol
        for ci in sindex["circle_grid"].query(x0, y0, x1, y1):
            if ci in seen_c:
                continue
            seen_c.add(ci)
            c = sindex["circles"][ci]
            d, sv, _, foot = project_on_polyline(c["center"], poly)
            if d <= scfg.station_tol and scfg.end_margin < sv < L - scfg.end_margin:
                out.append({"s": sv, "point": foot, "kind": "breakpoint", "source": c["id"]})
        u = eng.unit((q[0] - p[0], q[1] - p[1]))
        for li in sindex["line_grid"].query(x0, y0, x1, y1):
            o = sindex["lines"][li]
            a, b = o["p0"], o["p1"]
            v = eng.unit((b[0] - a[0], b[1] - a[1]))
            if abs(u[0] * v[0] + u[1] * v[1]) > math.cos(math.radians(30)):
                continue          # parallel lines are not stations
            hit = _touch_or_cross(a, b, p, q, scfg.station_tol)
            if hit is None or (li, id(p)) in seen_l:
                continue
            seen_l.add((li, id(p)))
            d, sv, _, foot = project_on_polyline(hit, poly)
            if scfg.end_margin < sv < L - scfg.end_margin:
                out.append({"s": sv, "point": foot, "kind": "line", "source": o["id"]})
    out.sort(key=lambda x: x["s"])
    merged: List[dict] = []
    for st in out:
        if merged and st["s"] - merged[-1]["s"] < scfg.station_merge:
            continue
        merged.append(st)
    return merged


def _touch_or_cross(a, b, p, q, tol) -> Optional[Tuple[float, float]]:
    """Point where segment ab touches (endpoint within tol) or crosses segment pq."""
    for e in (a, b):
        d, t = eng.point_segment_distance(e, p, q)
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


def _union_bbox(boxes: List[Sequence[float]]) -> Tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def group_annotations(texts: List[dict], scfg: Optional["SegmentConfig"] = None) -> List[dict]:
    """Group annotation keyword texts with their captions (CLAMP above a MARKER oval, ...)."""
    scfg = scfg or SegmentConfig()
    kw_texts = [t for t in texts if eng.annotation_keyword(t["norm"])]
    primaries = [t for t in kw_texts if eng.annotation_keyword(t["norm"]) in eng.ANNOTATION_INNER_WORDS]
    used = set()
    groups = []
    tindex = eng.build_text_index(texts, numeric_only=False)
    for p in primaries:
        d = p["dir"]
        pw = abs(d[0]) * (p["bbox"][2] - p["bbox"][0]) + abs(d[1]) * (p["bbox"][3] - p["bbox"][1])
        r = scfg.caption_max + pw
        near = [texts[ti] for ti in tindex.query(p["center"][0] - r, p["center"][1] - r,
                                                  p["center"][0] + r, p["center"][1] + r)]
        captions = []
        # captions sit above the keyword in its reading frame; vertical text may be written
        # in either direction, so for vertical text the opposite side is tried when the first
        # finds nothing (horizontal text reads one way only)
        for sign in ((1.0, -1.0) if abs(d[1]) > abs(d[0]) else (1.0,)):
            up = (sign * d[1], -sign * d[0])
            for t in near:
                if t["id"] == p["id"] or t["id"] in used:
                    continue
                vx, vy = t["center"][0] - p["center"][0], t["center"][1] - p["center"][1]
                h = vx * up[0] + vy * up[1]
                a = vx * d[0] + vy * d[1]
                tw = abs(d[0]) * (t["bbox"][2] - t["bbox"][0]) + abs(d[1]) * (t["bbox"][3] - t["bbox"][1])
                if scfg.caption_min < h < scfg.caption_max and abs(a) <= (pw + tw) / 2.0 + scfg.caption_min \
                        and eng.annotation_keyword(t["norm"]) not in eng.ANNOTATION_INNER_WORDS \
                        and not eng.is_generic_size(t["norm"]):      # 2M / 8M belong to terminals
                    captions.append((h, t))
            if captions:
                break
        captions.sort(key=lambda x: x[0])
        cap_texts = [t for _, t in captions]
        for t in cap_texts:
            used.add(t["id"])
        used.add(p["id"])
        kw = eng.annotation_keyword(p["norm"])
        cap_join = " ".join(t["norm"] for t in reversed(cap_texts))
        token = kw
        if kw == "MARKER" and any(eng.annotation_keyword(t["norm"]) == "CLAMP" for t in cap_texts):
            token = "CLAMP"
        groups.append({"token": token, "keyword_text_id": p["id"], "caption": cap_join or None,
                       "text_ids": [p["id"]] + [t["id"] for t in cap_texts],
                       "bbox": _union_bbox([p["bbox"]] + [t["bbox"] for t in cap_texts])})
    for t in kw_texts:
        if t["id"] in used:
            continue
        groups.append({"token": eng.annotation_keyword(t["norm"]), "keyword_text_id": t["id"], "caption": None,
                       "text_ids": [t["id"]], "bbox": tuple(t["bbox"])})
    return groups


def attach_annotations(ext: dict, segments: List[dict], sidx: "SegmentIndex",
                       scfg: SegmentConfig) -> Tuple[List[dict], List[dict]]:
    """Find where each annotation's leader touches a wire, then the owning segment."""
    groups = group_annotations(ext["texts"], scfg)
    by_drawing: Dict[int, List[dict]] = defaultdict(list)
    for o in ext["vectors"]["other_lines"]:            # leaders: any significant non-wire line
        if not eng.is_significant_line(o, ext["cfg"]):
            continue
        by_drawing[o["drawing_index"]].append(o)
    leaders = []
    lgrid = eng.GridIndex(25.0)
    for di, segs in sorted(by_drawing.items()):
        pts = [p for o in segs for p in (o["p0"], o["p1"])]
        ends = Counter((round(p[0], 2), round(p[1], 2)) for p in pts)
        if len(segs) >= 3 and all(n % 2 == 0 for n in ends.values()):
            continue                                    # closed outline (box, oval): not a leader
        leaders.append(pts)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        lgrid.insert(len(leaders) - 1, min(xs), min(ys), max(xs), max(ys))

    attached, node_level = [], []
    for g in groups:
        bb = g["bbox"]
        r = scfg.leader_text_reach
        best = None
        for li in lgrid.query(bb[0] - r, bb[1] - r, bb[2] + r, bb[3] + r):
            pts = leaders[li]
            head = min(pts, key=lambda p: eng.rect_distance(p, bb))
            hd = eng.rect_distance(head, bb)
            if hd > r:
                continue
            for foot_cand in pts:
                if eng.dist(foot_cand, head) < scfg.end_margin:
                    continue
                ns = sidx.nearest(foot_cand, scfg.leader_wire_tol)
                if ns and (best is None or hd < best[0]):
                    best = (hd, ns, "leader")
        if best is None:
            c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            ns = sidx.nearest(c, scfg.annotation_fallback_dist)
            if ns:
                best = (ns[0], ns, "proximity")
        if best is None:
            continue
        _, (d, seg, sv, foot), how = best
        L = polyline_length(seg["polyline"])
        rec = dict(g)
        rec["attach_point"] = foot
        rec["attach_method"] = how
        rec["confidence"] = 0.9 if how == "leader" else 0.5
        if sv <= scfg.node_annotation_tol or sv >= L - scfg.node_annotation_tol:
            rec["node"] = seg["start_node"] if sv <= scfg.node_annotation_tol else seg["end_node"]
            node_level.append(rec)
        else:
            rec["segment_index"] = seg["segment_index"]
            rec["s"] = sv
            attached.append(rec)
    return attached, node_level


def detect_inline_components(ext: dict, segments: List[dict], sidx: "SegmentIndex",
                             scfg: SegmentConfig) -> List[dict]:
    """Closed symbol shapes crossed by a wire inside a segment (connectors such as 7514VP)."""
    texts = ext["texts"]
    tindex = eng.build_text_index(texts, numeric_only=False)
    out = []
    for sh in ext["vectors"]["shapes"]:
        if sh["dashed"]:                       # any pen: closed outline crossed by a wire and named
            continue
        x0, y0, x1, y1 = sh["bbox"]
        size = max(x1 - x0, y1 - y0)
        if size < scfg.component_min_size or size > scfg.component_max_size:
            continue
        c = ((x0 + x1) / 2, (y0 + y1) / 2)
        for pos in sidx.segments_near_box(x0, y0, x1, y1):
            seg = segments[pos]
            poly = seg["polyline"]
            crossed = False
            for p, q in zip(poly, poly[1:]):
                d, _ = eng.point_segment_distance(c, p, q)
                if d <= min(x1 - x0, y1 - y0) / 2.0 + scfg.end_margin / 2.0:
                    crossed = True
                    break
            if not crossed:
                continue
            d, sv, _, foot = project_on_polyline(c, poly)
            L = polyline_length(poly)
            if sv < 3.0 * scfg.end_margin or sv > L - 3.0 * scfg.end_margin:
                continue
            r = scfg.component_text_radius
            best = None
            for ti in tindex.query(x0 - r, y0 - r, x1 + r, y1 + r):
                t = texts[ti]
                n = t["norm"]
                if t["is_numeric"] or eng.is_generic_size(n) or eng.annotation_keyword(n):
                    continue
                if not (eng.is_meaningful_identifier(n) or eng.is_spec_code(n)):
                    continue
                dd = eng.rect_distance(t["center"], (x0, y0, x1, y1))
                if dd <= r and (best is None or dd < best[0]):
                    best = (dd, t)
            if best is None:
                continue
            out.append({"segment_index": seg["segment_index"], "s": sv, "token": best[1]["norm"],
                        "text_ids": [best[1]["id"]], "shape_id": sh["id"], "attach_point": foot,
                        "confidence": 0.8})
    return out


def absorb_end_components(ext: dict, segments: List[dict], components: List[dict],
                          labels: Dict[int, List[dict]], scfg: SegmentConfig) -> List[dict]:
    """A named box crossed by the wire just before an unnamed wire end is that end's terminal.

    Typical case: an off-sheet reference box (B1) with the wire running on to the sheet edge.
    """
    nodes = ext["graph"]["nodes"]
    keep = []
    for c in components:
        seg = segments[c["segment_index"]]
        L = polyline_length(seg["polyline"])
        wires_s = [w["s"] for w in labels.get(seg["segment_index"], [])]
        absorbed = False
        for side, node_id, gap in (("start", seg["start_node"], c["s"]), ("end", seg["end_node"], L - c["s"])):
            n = nodes[node_id]
            if n["kind"] not in ("open", "terminal") or n.get("name") or gap > scfg.end_component_reach:
                continue
            between = [w for w in wires_s if (w < c["s"] if side == "start" else w > c["s"])]
            if between:
                continue
            n["kind"] = "terminal"
            n["name"] = canonical_name(c["token"])
            n["name_text_id"] = c["text_ids"][0]
            n["name_confidence"] = 0.6
            n["structural_id"] = f"TERMINAL[{n['name']}]"
            absorbed = True
            break
        if not absorbed:
            keep.append(c)
    return keep


# ---------------------------------------------------------------------------
# Initial (distance based) numeric label association
# ---------------------------------------------------------------------------

def initial_label_assignment(ext: dict, segments: List[dict], cfg: eng.EngineConfig,
                             scfg: SegmentConfig) -> Dict[int, List[dict]]:
    """Each numeric text goes to the nearest orientation-compatible wire piece.

    Either side is allowed, the side found by calibration being preferred.  This is the
    simple first-pass association; Script 3 replaces it by strict side rules and global
    one-to-one text ownership.
    """
    texts = ext["texts"]
    tindex = eng.build_text_index(texts, numeric_only=True)
    best: Dict[str, Tuple[float, int, int, float, float]] = {}
    for seg in segments:
        poly = seg["polyline"]
        cum = 0.0
        for pi, (a, b) in enumerate(zip(poly, poly[1:])):
            L = eng.dist(a, b)
            orient = eng.piece_orientation(a, b, cfg.axis_tol_deg)
            if orient in ("horizontal", "vertical") and L > 0:
                m = scfg.initial_label_max_dist + cfg.projection_margin
                for ti in tindex.query(min(a[0], b[0]) - m, min(a[1], b[1]) - m, max(a[0], b[0]) + m, max(a[1], b[1]) + m):
                    t = texts[ti]
                    if t["orientation"] != orient:
                        continue
                    cx, cy = t["center"]
                    if orient == "horizontal":
                        lo, hi = min(a[0], b[0]), max(a[0], b[0])
                        if not (lo - cfg.projection_margin <= cx <= hi + cfg.projection_margin):
                            continue
                    else:
                        lo, hi = min(a[1], b[1]), max(a[1], b[1])
                        if not (lo - cfg.projection_margin <= cy <= hi + cfg.projection_margin):
                            continue
                    d, tp = eng.point_segment_distance((cx, cy), a, b)
                    if d > scfg.initial_label_max_dist:
                        continue
                    conv = cfg.side_h if orient == "horizontal" else cfg.side_v
                    raw = (cy - (a[1] + b[1]) / 2) if orient == "horizontal" else (cx - (a[0] + b[0]) / 2)
                    rank = d if (conv == 0 or conv is None or conv * raw > 0) else d + 0.5 * scfg.initial_label_max_dist
                    key = t["id"]
                    cand = (rank, seg["segment_index"], pi, cum + tp * L, ti)
                    if key not in best or cand[0] < best[key][0]:
                        best[key] = cand
            cum += L
    per_seg: Dict[int, List[dict]] = defaultdict(list)
    for tid, (d, si, pi, sv, ti) in best.items():
        t = texts[ti]
        per_seg[si].append({"kind": "WIRE", "value": t["norm"], "s": sv, "text_id": tid,
                            "piece_index": pi, "distance": eng.rnd(d), "point": list(t["center"])})
    return per_seg


# ---------------------------------------------------------------------------
# Build segment JSON for one PDF
# ---------------------------------------------------------------------------

def endpoint_record(node: dict, wire_end: Sequence[float], annotations: List[dict], D) -> dict:
    rec = {
        "type": node["kind"],
        "structural_id": node.get("structural_id"),
        "node_id": f"N{node['id']}",
        "point": D(wire_end),
        "node_point": D(node["point"]),
        "symbol_id": node.get("symbol_id"),
        "symbol_kind": node.get("symbol_kind"),
    }
    if node["kind"] in ("junction", "branch"):
        rec["depth1_labels"] = list(node.get("depth1_labels") or [])
    if node["kind"] in ("terminal", "element", "open"):
        rec["name"] = node.get("name")
        rec["name_text_id"] = node.get("name_text_id")
        rec["name_confidence"] = node.get("name_confidence")
    rec["annotations"] = sorted(a["token"] for a in annotations)
    return rec


def segment_key(a: str, b: str) -> str:
    x, y = sorted([a or "?", b or "?"])
    return f"SEG[{x}<->{y}]"


def build_segments_for_pdf(pdf: str, page_no: int, cfg: eng.EngineConfig,
                          scfg: Optional[SegmentConfig] = None) -> Tuple[dict, dict]:
    """Return (segments JSON dict, extraction dict) for one PDF page.

    The drawing is calibrated first (Script 1); every distance used here is then scaled
    to that PDF's own units.  Work happens in the analysis frame; all coordinates written
    to the JSON are displayed-page coordinates.
    """
    doc = None
    try:
        doc, page = eng.open_pdf_page(pdf, page_no)
        ext = eng.extract_page(page, cfg)
        cfg = ext["cfg"]                                   # calibrated for this PDF
        scfg = scfg or SegmentConfig.scaled(cfg)
        info = eng.source_pdf_info(pdf, page)

        def D(p):                                          # analysis frame -> displayed page
            return eng.rpt(eng.to_display_pt(ext, p))
        assign_node_identities(ext, cfg, scfg)
        graph = ext["graph"]
        segs = trace_segments(graph, scfg.collinear_deg)
        sidx = SegmentIndex(segs)
        station_index = build_station_index(ext, scfg)
        annotations, node_annotations = attach_annotations(ext, segs, sidx, scfg)
        components = detect_inline_components(ext, segs, sidx, scfg)
        labels = initial_label_assignment(ext, segs, cfg, scfg)
        components = absorb_end_components(ext, segs, components, labels, scfg)
        node_ann: Dict[int, List[dict]] = defaultdict(list)
        for a in node_annotations:
            node_ann[a["node"]].append(a)
        seg_ann: Dict[int, List[dict]] = defaultdict(list)
        for a in annotations:
            seg_ann[a["segment_index"]].append(a)
        seg_comp: Dict[int, List[dict]] = defaultdict(list)
        for c in components:
            seg_comp[c["segment_index"]].append(c)

        out_segments = []
        nodes = graph["nodes"]
        for s in segs:
            si = s["segment_index"]
            sn, en = nodes[s["start_node"]], nodes[s["end_node"]]
            start = endpoint_record(sn, s["polyline"][0], node_ann.get(sn["id"], []), D)
            end = endpoint_record(en, s["polyline"][-1], node_ann.get(en["id"], []), D)
            names = {start.get("name"), end.get("name")}
            items = []
            for w in labels.get(si, []):
                items.append(dict(w, s=eng.rnd(w["s"]), point=D(w["point"])))
            for a in seg_ann.get(si, []):
                items.append({"kind": "ELEMENT", "value": a["token"], "s": eng.rnd(a["s"]), "annotation": True,
                              "caption": a["caption"], "text_ids": a["text_ids"],
                              "attach_method": a["attach_method"], "point": D(a["attach_point"])})
            for c in seg_comp.get(si, []):
                if c["token"] in names:
                    continue
                items.append({"kind": "ELEMENT", "value": c["token"], "s": eng.rnd(c["s"]), "annotation": False,
                              "component": True, "text_ids": c["text_ids"], "shape_id": c["shape_id"],
                              "point": D(c["attach_point"])})
            items.sort(key=lambda it: (it["s"], 0 if it["kind"] == "WIRE" else 1))
            sequence = [f"{it['kind']}:{it['value']}" for it in items]
            poly = s["polyline"]
            out_segments.append({
                "segment_index": si,
                "segment_key": segment_key(start["structural_id"], end["structural_id"]),
                "start": start,
                "end": end,
                "polyline": [D(p) for p in poly],
                "length": eng.rnd(polyline_length(poly)),
                "graph_points": [f"N{g}" for g in s["graph_points"]],
                "edge_ids": s["edge_ids"],
                "vector_line_ids": s["vector_line_ids"],
                "stations": [{"s": eng.rnd(st["s"]), "point": D(st["point"]), "kind": st["kind"],
                              "source": st["source"]} for st in detect_stations(poly, station_index, scfg)],
                "items": items,
                "sequence": sequence,
                "wire_numbers": [it["value"] for it in items if it["kind"] == "WIRE"],
            })

        node_json = []
        for n in nodes:
            if n["kind"] not in eng.BOUNDARY_KINDS:
                continue
            node_json.append({"node_id": f"N{n['id']}", "kind": n["kind"], "structural_id": n.get("structural_id"),
                              "point": D(n["point"]), "degree": len(n["edges"]),
                              "symbol_id": n.get("symbol_id"), "symbol_kind": n.get("symbol_kind"),
                              "name": n.get("name"), "name_text_id": n.get("name_text_id"),
                              "depth1_labels": n.get("depth1_labels"),
                              "annotations": sorted(a["token"] for a in node_ann.get(n["id"], []))})
        kinds: Dict[str, int] = defaultdict(int)
        for n in nodes:
            kinds[n["kind"]] += 1
        data = {
            "schema": SCHEMA,
            "generator": {"script": os.path.basename(__file__), "engine": eng.ENGINE_NAME,
                          "engine_version": eng.ENGINE_VERSION},
            "coordinate_frame": "displayed_page",
            "source_pdf": info,
            "config": {"engine": cfg.to_dict(), "segments": scfg.to_dict()},
            "profile": ext["profile"],
            "wire_width_used": ext["vectors"]["wire_width"],
            "summary": {
                "segments": len(out_segments),
                "wire_vectors": len(ext["vectors"]["wires"]),
                "texts": len(ext["texts"]),
                "numeric_texts": sum(1 for t in ext["texts"] if t["is_numeric"]),
                "node_kinds": dict(kinds),
                "junctions": sum(1 for n in nodes if n["kind"] == "junction"),
                "annotations_on_segments": len(annotations),
                "annotations_on_nodes": len(node_annotations),
                "inline_components": len(components),
                "sheet_frames": len(ext["border"]["frames"]),
                "border_texts_excluded": ext["border"]["border_texts_excluded"],
                "border_lines_excluded": ext["border"]["border_lines_excluded"],
                "table_components_dropped": len(ext["border"]["table_components_dropped"]),
                "bridged_gaps": graph.get("bridged_gaps", 0),
            },
            "sheet_border": ext["border"],
            "nodes": node_json,
            "junctions": [dict(j, point=D(j["point"]), directions=[dict(d, path=[D(p) for p in d["path"]])
                                                                   for d in j["directions"]])
                          for j in ext["junctions"]],
            "segments": out_segments,
        }
        return data, ext
    finally:
        if doc is not None:
            doc.close()


# ---------------------------------------------------------------------------
# Initial key-based comparison
# ---------------------------------------------------------------------------

def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        prev = cur
    return prev[-1]


def best_alignment(a: Sequence[str], b: Sequence[str]) -> Tuple[float, bool, int]:
    d_fwd = edit_distance(a, b)
    d_rev = edit_distance(a, list(reversed(b)))
    rev = d_rev < d_fwd
    d = d_rev if rev else d_fwd
    n = max(len(a), len(b))
    return (1.0 if n == 0 else 1.0 - d / n), rev, d


def initial_compare(seg_a: dict, seg_b: dict) -> dict:
    A = defaultdict(list)
    B = defaultdict(list)
    for s in seg_a["segments"]:
        A[s["segment_key"]].append(s)
    for s in seg_b["segments"]:
        B[s["segment_key"]].append(s)
    res = {"same": [], "changed": [], "removed": [], "added": []}
    for key in sorted(set(A) | set(B)):
        la, lb = list(A.get(key, [])), list(B.get(key, []))
        cands = []
        for i, sa in enumerate(la):
            for j, sb in enumerate(lb):
                sim, rev, d = best_alignment(sa["sequence"], sb["sequence"])
                cands.append((sim, -d, i, j, rev))
        cands.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
        ua, ub = set(), set()
        for sim, nd, i, j, rev in cands:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            rec = {"segment_key": key, "segment_a": la[i]["segment_index"], "segment_b": lb[j]["segment_index"],
                   "sequence_a": la[i]["sequence"], "sequence_b": lb[j]["sequence"],
                   "content_similarity": eng.rnd(sim, 4), "content_reversed": rev}
            res["same" if nd == 0 else "changed"].append(rec)
        for i, sa in enumerate(la):
            if i not in ua:
                res["removed"].append({"segment_key": key, "segment_a": sa["segment_index"], "sequence_a": sa["sequence"]})
        for j, sb in enumerate(lb):
            if j not in ub:
                res["added"].append({"segment_key": key, "segment_b": sb["segment_index"], "sequence_b": sb["sequence"]})
    res["summary"] = {"segments_a": len(seg_a["segments"]), "segments_b": len(seg_b["segments"]),
                      "same": len(res["same"]), "changed": len(res["changed"]),
                      "entire_branches_removed": len(res["removed"]), "entire_branches_added": len(res["added"])}
    return res


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def draw_route(img, poly, color, zoom, thick=None):
    pts = np.array([eng.zp(p, zoom) for p in poly], dtype=np.int32)
    if len(pts) < 2:
        return
    t = thick or max(3, int(round(1.6 * zoom)))
    cv2.polylines(img, [pts], False, (255, 255, 255), t + 4, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, color, t, cv2.LINE_AA)
    for p in (pts[0], pts[-1]):
        cv2.circle(img, tuple(int(v) for v in p), t + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in p), t + 1, color, 2, cv2.LINE_AA)


def render_overview(pdf: str, page_no: int, data: dict, zoom: float, path: str) -> None:
    doc = None
    try:
        doc, page = eng.open_pdf_page(pdf, page_no)
        img = eng.fade(eng.render_page_bgr(page, zoom), 0.5)
        for s in data["segments"]:
            col = eng.DIRECTION_COLORS[s["segment_index"] % len(eng.DIRECTION_COLORS)]
            draw_route(img, s["polyline"], col, zoom, thick=max(2, int(zoom)))
            poly = s["polyline"]
            mid = poly[len(poly) // 2] if len(poly) > 2 else ((poly[0][0] + poly[-1][0]) / 2, (poly[0][1] + poly[-1][1]) / 2)
            eng.put_label(img, str(s["segment_index"]), eng.zp(mid, zoom), (0, 0, 0), 0.4)
        cv2.imwrite(path, img)
    finally:
        if doc is not None:
            doc.close()


def render_initial_marks(pdf: str, page_no: int, data: dict, marks: List[Tuple[int, tuple]], zoom: float, path: str) -> None:
    doc = None
    try:
        doc, page = eng.open_pdf_page(pdf, page_no)
        img = eng.render_page_bgr(page, zoom)
        by_idx = {s["segment_index"]: s for s in data["segments"]}
        for idx, col in marks:
            draw_route(img, by_idx[idx]["polyline"], col, zoom)
        img = eng.add_legend_strip(img, [("changed", (0, 140, 255)), ("removed (PDF A)", (0, 0, 230)),
                                         ("added (PDF B)", (0, 170, 0))],
                                   title="Initial key-based comparison (Script 2)")
        cv2.imwrite(path, img)
    finally:
        if doc is not None:
            doc.close()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Script 2: connectivity graph and logical harness segments.")
    ap.add_argument("--pdf-a", required=True, help="Baseline / old PDF.")
    ap.add_argument("--pdf-b", help="New / comparison PDF (optional: segments for A only).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--page", type=int, default=0, help="Zero-based page number (default 0).")
    ap.add_argument("--zoom", type=float, default=3.0, help="Rendering zoom (default 3.0).")
    ap.add_argument("--no-images", action="store_true", help="Skip images.")
    eng.add_engine_args(ap)
    args = ap.parse_args(argv)
    if args.zoom <= 0:
        ap.error("--zoom must be positive")
    try:
        cfg = eng.engine_config_from_args(args)
        os.makedirs(args.out, exist_ok=True)
        written = []
        results = {}
        for tag, pdf in (("a", args.pdf_a), ("b", args.pdf_b)):
            if not pdf:
                continue
            data, _ = build_segments_for_pdf(pdf, args.page, cfg)
            p = os.path.join(args.out, f"pdf_{tag}_segments.json")
            write_json(p, data)
            written.append(p)
            results[tag] = data
            if not args.no_images:
                ov = os.path.join(args.out, f"pdf_{tag}_segments_overview.png")
                render_overview(pdf, args.page, data, args.zoom, ov)
                written.append(ov)
            s = data["summary"]
            print(f"PDF {tag.upper()}: {pdf}")
            print(f"   segments {s['segments']}, junctions {s['junctions']}, nodes {s['node_kinds']}, "
                  f"annotations {s['annotations_on_segments']}+{s['annotations_on_nodes']} (node), "
                  f"inline components {s['inline_components']}")
            pr = data["profile"]
            print(f"   calibration: wire pen {pr['wire_widths']}, text unit {pr['text_unit']}, dot {pr['dot_unit']}, "
                  f"de-skew {pr['deskew_deg']} deg, label side h={pr['side_h']} v={pr['side_v']}, "
                  f"label max {pr['label_max_side']}, endpoint tol {pr['endpoint_tol']}")
            print(f"   sheet border: {s['sheet_frames']} frame(s); excluded {s['border_lines_excluded']} line(s), "
                  f"{s['border_texts_excluded']} margin text(s), {s['table_components_dropped']} table component(s)")
        if "a" in results and "b" in results:
            cmp_res = initial_compare(results["a"], results["b"])
            p = os.path.join(args.out, "segment_comparison_initial.json")
            write_json(p, cmp_res)
            written.append(p)
            if not args.no_images:
                ma = [(r["segment_a"], (0, 140, 255)) for r in cmp_res["changed"]] + \
                     [(r["segment_a"], (0, 0, 230)) for r in cmp_res["removed"]]
                mb = [(r["segment_b"], (0, 140, 255)) for r in cmp_res["changed"]] + \
                     [(r["segment_b"], (0, 170, 0)) for r in cmp_res["added"]]
                pa = os.path.join(args.out, "pdf_a_segments_initial.png")
                pb = os.path.join(args.out, "pdf_b_segments_initial.png")
                render_initial_marks(args.pdf_a, args.page, results["a"], ma, args.zoom, pa)
                render_initial_marks(args.pdf_b, args.page, results["b"], mb, args.zoom, pb)
                written += [pa, pb]
            print(f"Initial comparison: {cmp_res['summary']}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for p in written:
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
