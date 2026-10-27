#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presentation-variation matrix for YOUR drawings.

    python tests/run_variations.py --pdf-a old.pdf [--pdf-b new.pdf] --out variation_runs
                                   [--only scale0.5,rot90] [--jobs 4] [--page 0]

Every variation redraws a PDF with presentation changes only (page scale, rotation, wire
stroke width, font size, label distance / side / angle, dot and circle size, wire ends
moved, wire vectors split into pieces).  For each variation V:

  * A versus V(A) must report no difference at all (0 changed / review / removed / added);
  * A versus V(B) must report the same counts as A versus B (when --pdf-b is given).

Exit code 0 when every variation behaves, 1 otherwise.  NOTE (not a failure): under the
variations that move labels (label shift, mirror, combined) a label of B that sits close to
the allowed label distance can legitimately move between "review" (uncertain association)
and "same", because the allowed distance is relative to each drawing's typical label
distance.  Only the review count may differ in that case.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VARIATIONS = {
    "identity": [],
    "scale0.5": ["--scale", "0.5"],
    "scale1.8": ["--scale", "1.8"],
    "wire2x": ["--wire-width", "2.0"],
    "wire0.5x": ["--wire-width", "0.5"],
    "font0.7": ["--font", "0.7"],
    "font1.4": ["--font", "1.4"],
    "labelshift": ["--label-shift", "0.55"],
    "mirror": ["--mirror-labels"],
    "jitter4deg": ["--text-jitter", "4"],
    "topdown": ["--topdown"],
    "dot2x": ["--dot", "2.0"],
    "circle0.7": ["--circle", "0.7"],
    "endjitter": ["--endpoint-jitter", "1.1"],
    "segment4": ["--segment", "4"],
    "skew7": ["--skew", "7"],
    "rot90": ["--skew", "90"],
    "combined": ["--scale", "0.7", "--wire-width", "1.6", "--font", "1.2", "--label-shift", "0.28",
                 "--text-jitter", "3", "--dot", "1.5", "--endpoint-jitter", "0.75", "--segment", "3", "--skew", "3"],
}
KEYS = ("changed", "review", "entire_branches_removed", "entire_branches_added")
LABEL_MOVING = {"labelshift", "mirror", "combined"}


def sh(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def compare(pdf_a: str, pdf_b: str, work: str, page: int):
    """Script 2 + Script 3 for one PDF pair; returns the summary dict or an error string."""
    seg, cmp_ = work + "_seg", work + "_cmp"
    rc, log = sh([os.path.join(ROOT, "build_logical_segments.py"), "--pdf-a", pdf_a, "--pdf-b", pdf_b,
                  "--out", seg, "--no-images", "--page", str(page)])
    if rc:
        return "Script 2 failed: " + (log.splitlines() or [""])[-1]
    rc, log = sh([os.path.join(ROOT, "compare_and_render_segments_v6.py"), "--pdf-a", pdf_a, "--pdf-b", pdf_b,
                  "--segments-a", os.path.join(seg, "pdf_a_segments.json"),
                  "--segments-b", os.path.join(seg, "pdf_b_segments.json"),
                  "--out", cmp_, "--zoom", "1", "--page", str(page)])
    if rc:
        return "Script 3 failed: " + (log.splitlines() or [""])[-1]
    with open(os.path.join(cmp_, "segment_comparison_v6.json"), encoding="utf-8") as fh:
        return json.load(fh)["summary"]


def fmt(s) -> str:
    if isinstance(s, str):
        return s
    return (f"same {s['same_exact'] + s['same_probable']:4d}  chg {s['changed']}  rev {s['review']}  "
            f"del {s['entire_branches_removed']}  add {s['entire_branches_added']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Presentation-variation matrix for a harness PDF pair.")
    ap.add_argument("--pdf-a", required=True)
    ap.add_argument("--pdf-b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma separated variation names (default: all)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--page", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in VARIATIONS if not a.only or n in a.only.split(",")]
    base = None
    if a.pdf_b:
        base = compare(a.pdf_a, a.pdf_b, os.path.join(a.out, "baseline"), a.page)
        print(f"baseline A vs B        : {fmt(base)}")
        if isinstance(base, str):
            return 1

    def job(name):
        res = {}
        for tag, src in (("a", a.pdf_a), ("b", a.pdf_b)):
            if src is None:
                continue
            vpdf = os.path.join(a.out, f"{name}_{tag}.pdf")
            rc, log = sh([os.path.join(HERE, "make_variation.py"), src, vpdf, "--page", str(a.page)] + VARIATIONS[name])
            if rc:
                res[tag] = "generator failed: " + (log.splitlines() or [""])[-1]
                continue
            res[tag] = compare(a.pdf_a, vpdf, os.path.join(a.out, f"{name}_{tag}"), 0)
        return name, res

    bad = notes = 0
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for name, res in ex.map(job, names):
            ra, rb = res.get("a"), res.get("b")
            ok_a = isinstance(ra, dict) and all(ra[k] == 0 for k in KEYS)
            ok_b = rb is None or (isinstance(rb, dict) and all(rb[k] == base[k] for k in KEYS))
            note = (ok_a and not ok_b and name in LABEL_MOVING and isinstance(rb, dict)
                    and all(rb[k] == base[k] for k in KEYS if k != "review"))
            status = "OK  " if ok_a and ok_b else ("NOTE" if note else "BAD ")
            bad += status == "BAD "
            notes += status == "NOTE"
            line = f"{status} {name:11s} | A vs V(A): {fmt(ra):38s}"
            if rb is not None:
                line += f" | A vs V(B): {fmt(rb)}"
            print(line, flush=True)
    print(f"{len(names) - bad}/{len(names)} variations behave"
          + (f" ({notes} NOTE: only the review count moved under a label-moving variation)" if notes else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
