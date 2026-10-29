#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end verification of the three harness-comparison scripts.

Builds two small synthetic harness PDFs that reproduce the known false-positive
situations, runs Script 2 and Script 3 through their command lines and checks
the results.

    python tests/verify_pipeline.py            (exit code 0 = all checks passed)

Synthetic drawing (PDF A), all wires 0.539 pt, junction = filled dot + ring:
    T_MAIN --105-- J1 --51-- J2 --52-- T_R
                   |          |\\
                  60 (vert)  32 \\ (angled, no label)
                   |          |   \\__41__CLAMP__ T_DIAG
                   +--61-- T_UP   J3 ... vertical bus with
                                   parallel branches 85 / 87 / 88 (8 pt apart)
                                   and 67 / 66 (4 pt apart, one visible 66)
    T_EN1 | T_EN2 vertical wire with EN6049-007 (left), 124 (left) and 123 (right)
    999 printed ABOVE the trunk (must never be used)
PDF B = same harness shifted by (15, 10), every wire drawn in reverse direction and
order, 52 -> 53 and 51 -> 57 (so junction J2 changes two of its three depth-1 labels),
the CLAMP annotation deleted, 105 nudged by 1 pt.
PDF C = PDF A plus a sheet border drawn with the WIRE stroke width (double frame, zone
ticks, zone numbers, a title-block table hanging on the frame) and a crop box that cuts
through the terminal name 7554VC-A.  A versus C must show no difference at all.
Presentation variants V(A), V(B) (tests/make_variation.py): page scale, rotation, wire
stroke width, font size, dot size, wire ends moved and wires split into pieces.  A versus
V(A) must show no difference; A versus V(B) must give the same findings as A versus B.
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import tempfile

import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["junction_depth1_ids.py", "build_logical_segments.py", "compare_and_render_segments_v6.py",
           "batch_compare.py", "harness_compare_app.py"]
FS = 5.39
FAILURES: list = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


# ---------------------------------------------------------------------------
# Synthetic drawing
# ---------------------------------------------------------------------------

def build_pdf(path: str, variant: str) -> None:
    dx, dy = (15.0, 10.0) if variant == "B" else (0.0, 0.0)
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    P = lambda x, y: fitz.Point(x + dx, y + dy)  # noqa: E731
    wires = []

    def wire(a, b):
        wires.append((a, b))

    def dot(x, y):
        page.draw_circle(P(x, y), 0.27, color=None, fill=(0, 0, 0), width=0)
        page.draw_circle(P(x, y), 0.67, color=(0, 0, 0), width=0.1)

    def box(x, y, name, side="left"):
        page.draw_rect(fitz.Rect(P(x - 3, y - 3), P(x + 3, y + 3)), color=(0, 0, 0), width=0.674)
        w = fitz.get_text_length(name, fontname="helv", fontsize=FS)
        tx = x - 5 - w if side == "left" else x + 5
        page.insert_text(P(tx, y), name, fontsize=FS, fontname="helv")
        page.insert_text(P(tx, y + 4.5), "2M", fontsize=4.05, fontname="helv")

    def htext(cx, cy, s):
        w = fitz.get_text_length(s, fontname="helv", fontsize=FS)
        page.insert_text(P(cx - w / 2, cy + FS * 0.35), s, fontsize=FS, fontname="helv")

    def vtext(cx, cy, s):
        w = fitz.get_text_length(s, fontname="helv", fontsize=FS)
        page.insert_text(P(cx + FS * 0.35, cy + w / 2), s, fontsize=FS, fontname="helv", rotate=90)

    # trunk T_MAIN - J1 - J2 - T_R
    box(40, 100, "7554VC-A")
    wire((43, 100), (100, 100))
    htext(72 - (1.0 if variant == "B" else 0.0), 105.5, "105")
    htext(72, 94.5, "999")                           # above the wire: invalid everywhere
    dot(100, 100)
    wire((100, 100), (200, 100))
    htext(150, 105.5, "57" if variant == "B" else "51")    # label inside both junction ids
    dot(200, 100)
    wire((200, 100), (257, 100))
    htext(228, 105.5, "53" if variant == "B" else "52")
    box(260, 100, "6301VS370-B", side="right")
    # J1 up branch: vertical 60 then horizontal 61 to T_UP
    wire((100, 100), (100, 60))
    vtext(104.5, 80, "60")
    dot(100, 60)
    wire((100, 60), (157, 60))
    htext(128, 65.5, "61")
    box(160, 60, "6711VN/B2", side="right")
    # J2 bent branch: angled piece then horizontal 41 (+ CLAMP in A) to T_DIAG
    wire((200, 100), (215, 125))
    wire((215, 125), (277, 125))
    htext(235, 130.5, "41")
    box(280, 125, "14MP-A", side="right")
    if variant in ("A", "C"):
        page.draw_line(P(258, 125), P(258, 116), color=(0, 0, 0), width=0.337)
        page.draw_oval(fitz.Rect(P(251, 111), P(265, 116)), color=(0, 0, 0), width=0.877)
        htext(258, 113.5, "MARKER")
        htext(258, 107.5, "CLAMP")
    # J2 down: vertical 32 to the bus junction J3
    wire((200, 100), (200, 150))
    vtext(204.5, 125, "32")
    # vertical bus with parallel branches
    ys = [150, 158, 166, 220, 224]
    labels = {150: ("85", 154.5), 158: ("87", 162.5), 166: ("88", 170.5)}
    for a, b in zip(ys, ys[1:]):
        wire((200, a), (200, b))
    for y in ys:
        dot(200, y)
    wire((200, 224), (200, 260))
    box(200, 263, "1990VC-A", side="right")
    vtext(204.5, 190, "14")
    for y in (150, 158, 166):
        wire((200, y), (277, y))
        htext(240, labels[y][1], labels[y][0])
    box(280, 150, "6709VN/DC2", side="right")
    box(280, 158, "6708VN/DC2", side="right")
    box(280, 166, "7502VT-A", side="right")
    wire((200, 220), (277, 220))
    htext(215, 225.5, "67")
    wire((200, 224), (277, 224))
    htext(245, 229.5, "66")                          # one visible 66 next to two branches
    box(280, 220, "7502VT-B", side="right")
    box(280, 226, "7502VT-C", side="right")
    vtext(204.5, 242, "20")
    # isolated vertical route: EN6049-007 and 124 on the left (invalid), 123 on the right
    box(400, 57, "2LK", side="right")
    wire((400, 60), (400, 140))
    vtext(395.5, 85, "EN6049-007")
    vtext(395.5, 120, "124")
    vtext(404.5, 100, "123")
    box(400, 143, "2LM", side="right")

    order = list(reversed(wires)) if variant == "B" else wires
    for a, b in order:
        if variant == "B":
            a, b = b, a
        page.draw_line(P(*a), P(*b), color=(0, 0, 0), width=0.539)
    if variant == "C":
        add_sheet_border(page, htext)
        page.set_cropbox(fitz.Rect(20, 0, 600, 400))   # cuts the first characters of 7554VC-A
    doc.save(path)
    doc.close()


def add_sheet_border(page, htext, width: float = 0.539) -> None:
    """Frame, zone ticks, zone numbers and a title-block table, all with the wire stroke width."""
    outer, inner = (1, 26, 594, 384), (5, 30, 590, 380)
    for r in (outer, inner):
        page.draw_rect(fitz.Rect(*r), color=(0, 0, 0), width=width)
    for k in range(8):
        x = outer[0] + k * (outer[2] - outer[0]) / 8
        if k:
            page.draw_line(fitz.Point(x, outer[1]), fitz.Point(x, inner[1]), color=(0, 0, 0), width=width)
            page.draw_line(fitz.Point(x, inner[3]), fitz.Point(x, outer[3]), color=(0, 0, 0), width=width)
        xc = x + (outer[2] - outer[0]) / 16
        for yc in (28.0, 382.0):
            w = fitz.get_text_length(str(8 - k), fontname="helv", fontsize=3.5)
            page.insert_text(fitz.Point(xc - w / 2, yc + 1.2), str(8 - k), fontsize=3.5, fontname="helv")
    for y in (330, 342, 354, 366):                     # title block rows end on the frame
        page.draw_line(fitz.Point(450, y), fitz.Point(inner[2], y), color=(0, 0, 0), width=width)
        htext(485, y + 5.5, str(100 + (y - 330) // 12))
    for x in (450, 520):                               # columns end on the frame
        page.draw_line(fitz.Point(x, 330), fitz.Point(x, inner[3]), color=(0, 0, 0), width=width)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run(cmd) -> subprocess.CompletedProcess:
    res = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
    return res


def seq_of(segments, a_name, b_name):
    for s in segments:
        ids = {s["start"].get("structural_id"), s["end"].get("structural_id")}
        if any(a_name in (i or "") for i in ids) and any(b_name in (i or "") for i in ids):
            return s
    return None


def same_either_way(seq, expected):
    return seq == expected or seq == list(reversed(expected))


def main() -> int:
    print("1. compile + source hygiene")
    for name in SCRIPTS:
        path = os.path.join(ROOT, name)
        try:
            py_compile.compile(path, doraise=True)
            ok = True
        except py_compile.PyCompileError as exc:
            ok = False
            print(exc)
        check(ok, f"{name} compiles")
        src = open(path, encoding="utf-8").read()
        amp, lt = chr(38), chr(60)
        markup = [lt + "br", lt + "strong", lt + "/strong", lt + "p>", lt + "b>", lt + "i>",
                  amp + "lt;", amp + "gt;", amp + "amp;", amp + "nbsp;", amp + "quot;"]
        bad = [m for m in markup if m in src]
        check(not bad, f"{name} has no HTML entities / markup")

    print("2. numeric regex")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    import compare_and_render_segments_v6 as v6  # noqa: E402
    for s, exp in (("52", True), ("52.0", True), ("35.8", True), ("EN6049-007", False), ("2M", False),
                   ("CLAMP", False), ("1 2", False), ("12a", False)):
        check(v6.is_numeric_text(v6.normalize_text(s)) == exp, f"numeric({s!r}) == {exp}")
    check(v6.normalize_text("52.0") == "52", "52.0 normalizes to 52")
    for s in ("2M", "8M", "2M,8M", "EN6049-007", "NSA935805-06", "CLAMP", "123"):
        check(not v6.is_meaningful_identifier(s), f"{s} is not a terminal identity")
    for s in ("7554VC-A", "6301VS370-B", "6307VS360-C", "6711VN/B2", "14MP-A"):
        check(v6.is_meaningful_identifier(s), f"{s} is a terminal identity")

    print("3. side rules follow the PDF y-down frame")
    cfg = v6.V6Config()
    hp = {"start": (0, 100), "end": (60, 100), "orientation": "horizontal"}
    vp = {"start": (100, 0), "end": (100, 60), "orientation": "vertical"}
    T = lambda c, o, n="52": {"is_numeric": True, "center": list(c), "orientation": o, "norm": n,  # noqa: E731
                              "bbox": [c[0] - 3, c[1] - 2.5, c[0] + 3, c[1] + 2.5]}
    check(v6.strict_label_metrics(hp, T((30, 105.5), "horizontal"), cfg)[0] is not None, "below horizontal wire accepted")
    check(v6.strict_label_metrics(hp, T((30, 94.5), "horizontal"), cfg)[1] == "wrong_side", "above horizontal wire rejected")
    check(v6.strict_label_metrics(hp, T((30, 105.5), "vertical"), cfg)[1] == "text_not_horizontal", "rotated text on horizontal wire rejected")
    check(v6.strict_label_metrics(hp, T((90, 105.5), "horizontal"), cfg)[1] == "outside_projection", "text beyond the piece end rejected")
    tv = lambda c: {"is_numeric": True, "center": list(c), "orientation": "vertical", "norm": "32",  # noqa: E731
                    "bbox": [c[0] - 2.5, c[1] - 3, c[0] + 2.5, c[1] + 3]}
    check(v6.strict_label_metrics(vp, tv((104.5, 30)), cfg)[0] is not None, "right of vertical wire accepted")
    check(v6.strict_label_metrics(vp, tv((95.5, 30)), cfg)[1] == "wrong_side", "left of vertical wire rejected")
    ap = {"start": (0, 0), "end": (30, 40), "orientation": "angled"}
    check(v6.strict_label_metrics(ap, T((20, 25), "horizontal"), cfg)[1] == "angled_piece", "angled piece claims nothing")
    s, rev, d = v6.align(["WIRE:163", "WIRE:1310", "ELEMENT:ROUTE", "WIRE:169", "ELEMENT:LABEL", "WIRE:198", "WIRE:150"],
                         ["WIRE:156", "WIRE:163", "WIRE:1310", "ELEMENT:ROUTE", "WIRE:169", "ELEMENT:LABEL", "WIRE:198", "WIRE:150"])
    check(d == 1 and not rev, "insertion at the start: direct orientation, edit distance 1")
    s, rev, d = v6.align(["WIRE:41", "WIRE:32"], ["WIRE:32", "WIRE:41"])
    check(d == 0 and rev, "[41, 32] vs [32, 41] identical when reversed")
    sim, info = v6.multiset_junction_similarity(["41", "65", "86"], ["41", "65", "UNLABELLED"], 2)
    check(info["related"] and info["unresolved"] == 1 and sim > 0.75, "DOT1[41|65|86] ~ DOT1[41|65|UNLABELLED]")
    sim, info = v6.multiset_junction_similarity(["45", "181", "242", "242", "245"], ["45", "181", "207", "242", "242"], 2)
    check(info["common"] == 4 and info["related"], "multiset junction ids share 4 of 5 values")

    print("4. end-to-end on synthetic PDFs")
    tmp = tempfile.mkdtemp(prefix="harness_verify_")
    pa, pb = os.path.join(tmp, "a.pdf"), os.path.join(tmp, "b.pdf")
    build_pdf(pa, "A")
    build_pdf(pb, "B")
    seg_dir, out_dir = os.path.join(tmp, "segments_pair"), os.path.join(tmp, "comparison_output")
    r = run([os.path.join(ROOT, "build_logical_segments.py"), "--pdf-a", pa, "--pdf-b", pb, "--out", seg_dir,
             "--no-images"])
    check(r.returncode == 0, "Script 2 runs")
    r = run([os.path.join(ROOT, "compare_and_render_segments_v6.py"), "--pdf-a", pa, "--pdf-b", pb,
             "--segments-a", os.path.join(seg_dir, "pdf_a_segments.json"),
             "--segments-b", os.path.join(seg_dir, "pdf_b_segments.json"), "--out", out_dir, "--zoom", "2"])
    check(r.returncode == 0, "Script 3 runs")
    if r.returncode != 0:
        return 1
    for f in ("pdf_a_segments_corrected_v6.json", "pdf_b_segments_corrected_v6.json", "segment_comparison_v6.json",
              "pdf_a_segment_changes_v6.png", "pdf_b_segment_changes_v6.png"):
        check(os.path.isfile(os.path.join(out_dir, f)) and os.path.getsize(os.path.join(out_dir, f)) > 0, f"{f} written")
    ca = json.load(open(os.path.join(out_dir, "pdf_a_segments_corrected_v6.json")))
    cmp_ = json.load(open(os.path.join(out_dir, "segment_comparison_v6.json")))
    segs = ca["segments"]
    expect = [("7554VC-A", "DOT1", ["WIRE:105"]), ("6711VN/B2", "DOT1", ["WIRE:60", "WIRE:61"]),
              ("6301VS370-B", "DOT1", ["WIRE:52"]), ("6709VN/DC2", "DOT1", ["WIRE:85"]),
              ("6708VN/DC2", "DOT1", ["WIRE:87"]), ("7502VT-A", "DOT1", ["WIRE:88"]),
              ("7502VT-B", "DOT1", ["WIRE:67"]), ("7502VT-C", "DOT1", ["WIRE:66"]),
              ("2LK", "2LM", ["WIRE:123"]), ("14MP-A", "DOT1", ["WIRE:41", "ELEMENT:CLAMP"])]
    for a, b, exp in expect:
        s = seq_of(segs, a, b)
        check(s is not None and same_either_way(s["sequence"], exp),
              f"segment {a}: {exp}  (got {s['sequence'] if s else None})")
    all_ids = [x["text_id"] for s in segs for x in s["numeric_text_assignments"]]
    check(len(all_ids) == len(set(all_ids)), "every numeric text ID owned once (global ownership)")
    values = [x["value"] for s in segs for x in s["numeric_text_assignments"]]
    check(values.count("66") == 1, "the single visible 66 is used exactly once")
    check("999" not in values and "124" not in values, "labels above / left of a wire are never used")
    check(not any("EN6049" in t for s in segs for t in s["sequence"]), "EN6049-007 never becomes a wire number")
    term_ids = [ep["identity"]["text_id"] for s in segs for ep in (s["start"], s["end"])
                if isinstance(ep.get("identity"), dict) and ep["identity"].get("text_id")]
    nodes = {}
    for s in segs:
        for ep in (s["start"], s["end"]):
            tid = (ep.get("identity") or {}).get("text_id") if isinstance(ep.get("identity"), dict) else None
            if tid:
                nodes.setdefault(tid, set()).add(tuple(ep["node_point"]))
    check(all(len(v) == 1 for v in nodes.values()) and term_ids, "every terminal text ID identifies one endpoint")
    check(not any(ep.get("structural_id") in ("TERMINAL[2M]", "TERMINAL[8M]") for s in segs for ep in (s["start"], s["end"])),
          "generic 2M / 8M never used as terminal identity")
    j1 = [ep for s in segs for ep in (s["start"], s["end"]) if ep.get("structural_id") == "DOT1[51|60|105]"]
    check(bool(j1), "105 kept both as junction label and as first wire number (DOT1[51|60|105])")
    sm = cmp_["summary"]
    changes = [c for r in cmp_["changed"] for c in r["changes"]]
    check(sm["changed"] == 2 and "WIRE:52 changed to WIRE:53" in changes, "52 -> 53 reported as change")
    check("WIRE:51 changed to WIRE:57" in changes,
          "51 -> 57 between two junctions whose ids both changed is a change, not removed + added")
    check(sm["review"] == 1 and cmp_["review"][0]["review_type"] == "annotation_only", "deleted CLAMP is an annotation review")
    check(sm["entire_branches_removed"] == 0 and sm["entire_branches_added"] == 0,
          "no false branch removal / addition (shift, reverse traversal, nudged 105)")
    check(sm["same_exact"] + sm["same_probable"] == sm["segments_a"] - 3, "all other segments are the same")
    print("5. sheet border at wire width + crop through a terminal name (A vs C)")
    pc = os.path.join(tmp, "c.pdf")
    build_pdf(pc, "C")
    seg_c, out_c = os.path.join(tmp, "segments_border"), os.path.join(tmp, "comparison_border")
    r = run([os.path.join(ROOT, "build_logical_segments.py"), "--pdf-a", pa, "--pdf-b", pc, "--out", seg_c,
             "--no-images"])
    check(r.returncode == 0, "Script 2 runs on the bordered PDF")
    r = run([os.path.join(ROOT, "compare_and_render_segments_v6.py"), "--pdf-a", pa, "--pdf-b", pc,
             "--segments-a", os.path.join(seg_c, "pdf_a_segments.json"),
             "--segments-b", os.path.join(seg_c, "pdf_b_segments.json"), "--out", out_c, "--zoom", "2"])
    check(r.returncode == 0, "Script 3 runs on the bordered PDF")
    if r.returncode == 0:
        sc = json.load(open(os.path.join(seg_c, "pdf_b_segments.json")))
        cc = json.load(open(os.path.join(out_c, "segment_comparison_v6.json")))
        check(sc["sheet_border"]["drawing_area"] is not None, "sheet frame detected in the bordered PDF")
        check(len(sc["sheet_border"]["table_components_dropped"]) >= 1, "title-block table dropped")
        check(sc["summary"]["segments"] == json.load(open(os.path.join(seg_c, "pdf_a_segments.json")))["summary"]["segments"],
              "bordered PDF has exactly the harness segments of the plain PDF")
        sm = cc["summary"]
        check(sm["changed"] == 0 and sm["review"] == 0 and sm["entire_branches_removed"] == 0
              and sm["entire_branches_added"] == 0, f"border creates no findings (summary {sm['changed']}/{sm['review']}/"
              f"{sm['entire_branches_removed']}/{sm['entire_branches_added']})")
        check(sm["same_exact"] == sm["segments_a"], "every segment is same exact")
        names = {ep.get("name") for s in sc["segments"] for ep in (s["start"], s["end"])}
        check("7554VC-A" in names, "terminal name cut by the crop box is read in full")
    print("6. presentation variations (scale, rotation, pen width, font, dots, segmentation)")
    from make_variation import make_variation  # noqa: E402
    variants = {
        "scaled + skewed": dict(scale=0.6, skew=4.0, wire_width=2.2, font=1.15, dot=1.8, segment=3,
                                endpoint_jitter=0.5, seed=3),
        "turned 90 deg": dict(scale=1.5, skew=90.0, wire_width=0.45, font=0.85, segment=2, seed=5),
    }
    for vname, opts in variants.items():
        for tag, src, exp in (("a", pa, (0, 0, 0, 0)), ("b", pb, (2, 1, 0, 0))):
            vp = os.path.join(tmp, f"var_{vname.split()[0]}_{tag}.pdf")
            make_variation(src, vp, **opts)
            seg_v, out_v = vp[:-4] + "_seg", vp[:-4] + "_cmp"
            r2 = run([os.path.join(ROOT, "build_logical_segments.py"), "--pdf-a", pa, "--pdf-b", vp, "--out", seg_v,
                      "--no-images"])
            r3 = run([os.path.join(ROOT, "compare_and_render_segments_v6.py"), "--pdf-a", pa, "--pdf-b", vp,
                      "--segments-a", os.path.join(seg_v, "pdf_a_segments.json"),
                      "--segments-b", os.path.join(seg_v, "pdf_b_segments.json"), "--out", out_v, "--zoom", "1"]) \
                if r2.returncode == 0 else r2
            got = None
            if r3.returncode == 0:
                sm = json.load(open(os.path.join(out_v, "segment_comparison_v6.json")))["summary"]
                got = (sm["changed"], sm["review"], sm["entire_branches_removed"], sm["entire_branches_added"])
            what = "no difference" if tag == "a" else "same findings as A vs B"
            check(got == exp, f"{vname}: A vs V({tag.upper()}) gives {what} (changed/review/removed/added {got})")
    print("7. multi-page PDFs (page pairing, pages without a harness, best match)")
    import batch_compare  # noqa: E402
    check(batch_compare.plan_pairs(1, 3) == ([(0, 0), (0, 1), (0, 2)], [], []), "auto: one page against every page")
    check(batch_compare.plan_pairs(3, 1) == ([(0, 0), (1, 0), (2, 0)], [], []), "auto: every page against the one page")
    check(batch_compare.plan_pairs(3, 2) == ([(0, 0), (1, 1)], [2], []), "auto: page by page, extra page only in A")
    check(batch_compare.plan_pairs(2, 2, "all-pairs") == ([(0, 0), (0, 1), (1, 0), (1, 1)], [], []),
          "all-pairs: every page against every page")
    multi = os.path.join(tmp, "multi.pdf")                 # [B, A, a page without a harness]
    doc = fitz.open()
    for src in (pb, pa):
        with fitz.open(src) as d:
            doc.insert_pdf(d)
    doc.new_page(width=600, height=400).insert_text((50, 80), "PARTS LIST - no harness on this page", fontsize=12)
    doc.save(multi)
    doc.close()
    res = batch_compare.run_batch(pa, multi, os.path.join(tmp, "batch"), "auto", 1.0, log=lambda s: None)
    rows = {(r["page_a"], r["page_b"]): r for r in res["pairs"]}
    check(set(rows) == {(1, 1), (1, 2), (1, 3)}, "the single page is compared with all 3 pages")
    r = rows.get((1, 2), {})
    check(r.get("status") == "no differences" and bool(r.get("best_match")), "the identical page is the best match")
    r = rows.get((1, 1), {})
    check(r.get("status") == "differences" and (r.get("summary") or {}).get("changed") == 2,
          "the edited page shows its changes")
    check(rows.get((1, 3), {}).get("status") == "skipped", "a page without a harness is skipped, the batch goes on")
    check(os.path.isfile(os.path.join(tmp, "batch", "batch_summary.csv")), "batch summary written")

    print("8. an endpoint terminal identity change is never hidden inside 'same'")

    def star(path, name, far=False):
        """A junction with three legs; the top leg ends at a box terminal called `name`."""
        d = fitz.open()
        p = d.new_page(width=400, height=300)
        J = (200, 150)
        for end, wire in (((200, 60), "100"), ((90, 150), "101"), ((310, 150), "102")):
            p.draw_line(fitz.Point(*J), fitz.Point(*end), color=(0, 0, 0), width=0.539)
        p.draw_circle(fitz.Point(*J), 0.27, color=None, fill=(0, 0, 0), width=0)
        p.draw_circle(fitz.Point(*J), 0.67, color=(0, 0, 0), width=0.1)
        for x, s in ((150, "101"), (255, "102")):
            p.insert_text((x, 146), s, fontsize=FS, fontname="helv")
        p.insert_text((210, 112), "100", fontsize=FS, fontname="helv")
        for bx, nm in ((84, "6700VN-B"), (316, "6701VN-C")):
            p.draw_rect(fitz.Rect(bx - 3, 147, bx + 3, 153), color=(0, 0, 0), width=0.674)
            p.insert_text((bx - fitz.get_text_length(nm, fontname="helv", fontsize=FS) / 2, 142), nm,
                          fontsize=FS, fontname="helv")
        p.draw_rect(fitz.Rect(197, 51, 203, 57), color=(0, 0, 0), width=0.674)          # top terminal box
        p.insert_text((200 - fitz.get_text_length(name, fontname="helv", fontsize=FS) / 2,
                       34 if far else 44), name, fontsize=FS, fontname="helv")           # far = low confidence
        d.save(path)
        d.close()

    def compare_star(nameb, far):
        ea, eb = os.path.join(tmp, "epA.pdf"), os.path.join(tmp, f"epB_{nameb}_{far}.pdf")
        star(ea, "7520VC-A")
        star(eb, nameb, far=far)
        sd, od = os.path.join(tmp, f"eps_{nameb}_{far}"), os.path.join(tmp, f"epc_{nameb}_{far}")
        run([os.path.join(ROOT, "build_logical_segments.py"), "--pdf-a", ea, "--pdf-b", eb, "--out", sd, "--no-images"])
        run([os.path.join(ROOT, "compare_and_render_segments_v6.py"), "--pdf-a", ea, "--pdf-b", eb,
             "--segments-a", os.path.join(sd, "pdf_a_segments.json"),
             "--segments-b", os.path.join(sd, "pdf_b_segments.json"), "--out", od, "--zoom", "1"])
        return json.load(open(os.path.join(od, "segment_comparison_v6.json")))

    cc = compare_star("3LJ", far=False)                     # clear name change -> changed
    sm = cc["summary"]
    changes = [c for r in cc["changed"] for c in r["changes"]]
    check(sm["changed"] == 1 and any("7520VC-A" in c and "3LJ" in c for c in changes),
          "a confident endpoint rename (7520VC-A -> 3LJ) is reported as a change")
    cl = compare_star("3LJ", far=True)                      # low-confidence name -> review, NOT same
    sm = cl["summary"]
    rtypes = {r.get("review_type") for r in cl["review"]}
    check(sm["changed"] + sm["review"] >= 1 and "endpoint_identity" in rtypes,
          "a low-confidence endpoint rename is surfaced for review, not hidden in 'same'")
    on_same = any("7520VC-A" in json.dumps(r) and "3LJ" in json.dumps(r)
                  for r in cl["same"] + cl["probable_same"])
    check(not on_same, "the renamed endpoint is not buried in a same / probable-same row")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED (work dir {tmp})")
        return 1
    print(f"All checks passed (work dir {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
