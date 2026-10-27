#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redraw a harness PDF with presentation variations only (no electrical change).

    python tests/make_variation.py SRC.pdf OUT.pdf [options]

  --scale k              page scale (geometry, stroke widths, fonts)
  --skew deg             whole drawing rotated by an arbitrary angle (90 = sheet turned)
  --wire-width f         harness wire stroke width x f (wire pen only)
  --font f               every text x f about its centre
  --label-shift d        numeric labels moved d text units further from their wire
  --mirror-labels        numeric labels moved to the other side of their wire
  --text-jitter deg      every text rotated by a random angle in [-deg, deg]
  --topdown              vertical texts read top-to-bottom instead of bottom-to-top
  --dot f                junction dots (small filled circles + rings) x f
  --circle f             large circles x f about their centre
  --endpoint-jitter j    wire ends randomly moved up to j wire widths (gaps / overshoots)
  --segment n            every wire split into up to n pieces, random order / direction,
                         small overlaps and duplicates
  --seed s

The source wire pen(s) and text unit are measured with Script 1's calibration, so any
harness PDF can be used.  Fonts are replaced by Helvetica.  The output page is unrotated.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import junction_depth1_ids as eng  # noqa: E402


def make_variation(src: str, out_path: str, scale: float = 1.0, skew: float = 0.0, wire_width: float = 1.0,
                   font: float = 1.0, label_shift: float = 0.0, mirror_labels: bool = False,
                   text_jitter: float = 0.0, topdown: bool = False, dot: float = 1.0, circle: float = 1.0,
                   endpoint_jitter: float = 0.0, segment: int = 0, seed: int = 1, page_no: int = 0) -> None:
    rng = random.Random(seed)
    sdoc = pymupdf.open(src)
    sp = sdoc[page_no]
    prof = eng.calibrate_profile(sp, eng.EngineConfig())["profile"]
    pens = prof["wire_widths"]
    U = prof["text_unit"]                          # source text unit (pt)
    M = sp.rotation_matrix
    W, H = sp.rect.width, sp.rect.height
    k = scale
    th = math.radians(skew)
    pad = abs(math.sin(th)) * max(W, H) * k
    NW, NH = W * k + 2 * pad, H * k + 2 * pad
    cx, cy = W / 2, H / 2

    def is_pen(w: float) -> bool:
        return any(abs(w - p) <= max(0.02, 0.06 * p) for p in pens)

    def G(p):
        """displayed source point -> output point"""
        x, y = (p[0] - cx) * k, (p[1] - cy) * k
        return pymupdf.Point(x * math.cos(th) - y * math.sin(th) + NW / 2,
                             x * math.sin(th) + y * math.cos(th) + NH / 2)

    def V(p):
        q = pymupdf.Point(p) * M
        return (q.x, q.y)

    page_rect = pymupdf.Rect(0, 0, W, H)

    def overlaps_page(r):
        """Degenerate (zero-width / zero-height) boxes of straight lines count too."""
        return r.x1 >= page_rect.x0 and r.x0 <= page_rect.x1 and r.y1 >= page_rect.y0 and r.y0 <= page_rect.y1

    def clip(p0, p1):
        """Clip a displayed segment to the visible source page (Liang-Barsky)."""
        x0, y0 = p0
        dx, dy = p1[0] - x0, p1[1] - y0
        t0, t1 = 0.0, 1.0
        for pp, qq in ((-dx, x0 - page_rect.x0), (dx, page_rect.x1 - x0), (-dy, y0 - page_rect.y0),
                       (dy, page_rect.y1 - y0)):
            if abs(pp) < 1e-12:
                if qq < 0:
                    return None
                continue
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
        return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)

    def roundish(d):
        items = d["items"]
        return all(it[0] in ("c", "l") for it in items) and sum(it[0] == "c" for it in items) >= 2

    def small_circle(d):
        r = pymupdf.Rect(d["rect"]) * M
        return roundish(d) and 0.02 * U < r.width <= 0.56 * U and r.height <= 0.56 * U

    def large_circle(d):
        r = pymupdf.Rect(d["rect"]) * M
        return roundish(d) and r.width > 0.74 * U and abs(r.width - r.height) < 0.2 * r.width

    drawings = sp.get_drawings()
    wires = []
    for d in drawings:
        if is_pen(d.get("width") or 0) and "s" in (d.get("type") or ""):
            wires += [(V(it[1]), V(it[2])) for it in d["items"] if it[0] == "l"]

    out = pymupdf.open()
    page = out.new_page(width=NW, height=NH)
    shape = page.new_shape()

    def finish(d, width):
        shape.finish(fill=d.get("fill"), color=d.get("color"), dashes=d.get("dashes"),
                     even_odd=d.get("even_odd", True), closePath=d.get("closePath", False),
                     lineJoin=int(d.get("lineJoin") or 0), lineCap=max(d.get("lineCap") or (0,)),
                     width=width, stroke_opacity=d.get("stroke_opacity") or 1, fill_opacity=d.get("fill_opacity") or 1)

    for d in drawings:
        if not overlaps_page(pymupdf.Rect(d["rect"]) * M):
            continue                                   # other sheets placed off the page
        w = d.get("width") or 0
        if is_pen(w) and "s" in (d.get("type") or "") and all(it[0] == "l" for it in d["items"]):
            for it in d["items"]:
                cl = clip(V(it[1]), V(it[2]))
                if cl is None:
                    continue
                p0, p1 = cl
                L = math.dist(p0, p1)
                ux, uy = ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L) if L else (0, 0)
                j = endpoint_jitter * w
                if j:
                    e0, e1 = rng.uniform(-j, j), rng.uniform(-j, j)
                    n0, n1 = rng.uniform(-j / 3, j / 3), rng.uniform(-j / 3, j / 3)
                    p0 = (p0[0] - ux * e0 - uy * n0, p0[1] - uy * e0 + ux * n0)
                    p1 = (p1[0] + ux * e1 - uy * n1, p1[1] + uy * e1 + ux * n1)
                pieces = [(p0, p1)]
                if segment and L > 1.1 * U:
                    n = rng.randint(2, segment)
                    ts = sorted(rng.uniform(0.05, 0.95) for _ in range(n - 1))
                    pts = [p0] + [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t) for t in ts] + [p1]
                    pieces = []
                    for q0, q1 in zip(pts, pts[1:]):
                        ov = rng.uniform(0, 0.55 * w)          # small overlap with the next piece
                        q1 = (q1[0] + ux * ov, q1[1] + uy * ov) if q1 != p1 else q1
                        pieces.append((q0, q1) if rng.random() < 0.5 else (q1, q0))
                    if rng.random() < 0.3:
                        pieces.append(rng.choice(pieces))      # duplicated piece
                    rng.shuffle(pieces)
                for q0, q1 in pieces:
                    shape.draw_line(G(q0), G(q1))
                    finish(d, w * k * wire_width)
            continue
        factor, centre = 1.0, None
        if (dot != 1.0 and small_circle(d)) or (circle != 1.0 and large_circle(d)):
            r = pymupdf.Rect(d["rect"]) * M
            centre = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            factor = dot if small_circle(d) else circle

        def T(p, centre=centre, factor=factor):
            q = V(p)
            if centre is not None:
                q = (centre[0] + (q[0] - centre[0]) * factor, centre[1] + (q[1] - centre[1]) * factor)
            return G(q)
        for it in d["items"]:
            if it[0] == "l":
                shape.draw_line(T(it[1]), T(it[2]))
            elif it[0] == "c":
                shape.draw_bezier(T(it[1]), T(it[2]), T(it[3]), T(it[4]))
            elif it[0] == "re":
                r = pymupdf.Rect(it[1])
                shape.draw_polyline([T(r.tl), T(r.tr), T(r.br), T(r.bl), T(r.tl)])
            elif it[0] == "qu":
                q = it[1]
                shape.draw_polyline([T(q.ul), T(q.ur), T(q.lr), T(q.ll), T(q.ul)])
        finish(d, w * k)
    shape.commit()

    def nearest_wire(c, horizontal):
        """Distance from a label centre to the nearest parallel wire on the label's usual side."""
        best = None
        lim, slack, flat = 2.2 * U, 0.37 * U, 0.1 * U
        for p0, p1 in wires:
            if horizontal and abs(p0[1] - p1[1]) < flat and min(p0[0], p1[0]) - slack <= c[0] <= max(p0[0], p1[0]) + slack:
                dd = c[1] - (p0[1] + p1[1]) / 2
            elif not horizontal and abs(p0[0] - p1[0]) < flat and min(p0[1], p1[1]) - slack <= c[1] <= max(p0[1], p1[1]) + slack:
                dd = c[0] - (p0[0] + p1[0]) / 2
            else:
                continue
            if 0 < dd < lim and (best is None or dd < best):
                best = dd
        return best

    for b in sp.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            ddx, ddy = line["dir"]
            dv = (M.a * ddx + M.c * ddy, M.b * ddx + M.d * ddy)
            for s in line["spans"]:
                txt = s["text"]
                if not txt.strip():
                    continue
                o = V(s["origin"])
                bb = pymupdf.Rect(s["bbox"]) * M
                if not overlaps_page(bb):
                    continue
                cen = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
                size = s["size"]
                ang = math.degrees(math.atan2(dv[1], dv[0]))
                horizontal, vertical = abs(dv[1]) < 0.2, abs(dv[0]) < 0.2
                if eng.is_numeric_text(eng.normalize_text(txt)) and (horizontal or vertical):
                    shift = label_shift * U
                    if mirror_labels:
                        dd = nearest_wire(cen, horizontal)
                        if dd is not None:
                            shift = -2 * dd
                    o = (o[0], o[1] + shift) if horizontal else (o[0] + shift, o[1])
                    cen = (cen[0], cen[1] + shift) if horizontal else (cen[0] + shift, cen[1])
                if font != 1.0:
                    o = (cen[0] + (o[0] - cen[0]) * font, cen[1] + (o[1] - cen[1]) * font)
                    size *= font
                if topdown and vertical and dv[1] < 0:
                    # same box, read top-to-bottom: baseline on the other side, start at the top
                    w_txt = pymupdf.get_text_length(txt, fontname="helv", fontsize=size)
                    o = (cen[0] - size * 0.35, cen[1] - w_txt / 2)
                    ang = 90.0
                ang += rng.uniform(-text_jitter, text_jitter) if text_jitter else 0.0
                pnt = G(o)
                c = s["color"]
                page.insert_text(pnt, txt, fontsize=size * k, fontname="helv",
                                 color=((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255),
                                 morph=(pnt, pymupdf.Matrix(-(ang + skew))))
    out.save(out_path, garbage=3, deflate=True)
    out.close()
    sdoc.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--wire-width", type=float, default=1.0)
    ap.add_argument("--font", type=float, default=1.0)
    ap.add_argument("--label-shift", type=float, default=0.0)
    ap.add_argument("--mirror-labels", action="store_true")
    ap.add_argument("--text-jitter", type=float, default=0.0)
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--dot", type=float, default=1.0)
    ap.add_argument("--circle", type=float, default=1.0)
    ap.add_argument("--endpoint-jitter", type=float, default=0.0)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    make_variation(a.src, a.out, scale=a.scale, skew=a.skew, wire_width=a.wire_width, font=a.font,
                   label_shift=a.label_shift, mirror_labels=a.mirror_labels, text_jitter=a.text_jitter,
                   topdown=a.topdown, dot=a.dot, circle=a.circle, endpoint_jitter=a.endpoint_jitter,
                   segment=a.segment, seed=a.seed, page_no=a.page)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
