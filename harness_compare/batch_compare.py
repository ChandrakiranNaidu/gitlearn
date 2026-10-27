#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two harness PDFs page by page (single-page or multi-page PDFs).

    python batch_compare.py --pdf-a old.pdf --pdf-b new.pdf --out results
                            [--pairing auto|page-by-page|all-pairs] [--zoom 2.0] [--mark-same]

Page pairing
  auto          One PDF has a single page and the other several: that page is compared
                with every page of the other PDF. Otherwise page by page (1-1, 2-2, ...);
                pages without a partner are listed as only in A / only in B.
  page-by-page  1-1, 2-2, ... always.
  all-pairs     Every page of A against every page of B (use when the page order differs).

Whenever a page is compared with several pages, the partner with the most matching
segments is marked as its best match.

Outputs in --out:
  A_p01_vs_B_p01/ ...       Script 3 results of each page pair (JSON, marked images, log)
  segments/                 Script 2 segment JSON of every page (computed once per page)
  batch_summary.json/.csv   one row per page pair

Pages without a harness drawing (cover sheet, parts list, ...) are reported as skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import threading
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)      # "fitz" import alias notice
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pymupdf  # noqa: E402

import build_logical_segments as seg_builder  # noqa: E402  (Script 2)
import compare_and_render_segments_v6 as v6  # noqa: E402  (Script 3)
import junction_depth1_ids as eng  # noqa: E402  (Script 1)

PAIRINGS = ("auto", "page-by-page", "all-pairs")
SUMMARY_KEYS = ("segments_a", "segments_b", "same_exact", "same_probable", "changed", "review",
                "entire_branches_removed", "entire_branches_added")


class Cancelled(Exception):
    pass


def page_count(pdf: str) -> int:
    with pymupdf.open(pdf) as doc:
        return doc.page_count


def plan_pairs(n_a: int, n_b: int, pairing: str = "auto") -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Zero-based (page A, page B) pairs plus pages of A / B that have no partner."""
    if pairing not in PAIRINGS:
        raise ValueError(f"unknown pairing {pairing!r} (use one of {', '.join(PAIRINGS)})")
    if pairing == "all-pairs":
        return [(i, j) for i in range(n_a) for j in range(n_b)], [], []
    if pairing == "auto" and n_a == 1 and n_b > 1:
        return [(0, j) for j in range(n_b)], [], []
    if pairing == "auto" and n_b == 1 and n_a > 1:
        return [(i, 0) for i in range(n_a)], [], []
    n = min(n_a, n_b)
    return [(i, i) for i in range(n)], list(range(n, n_a)), list(range(n, n_b))


def pair_folder(pa: int, pb: int) -> str:
    return f"A_p{pa + 1:02d}_vs_B_p{pb + 1:02d}"


def default_engine_config() -> "eng.EngineConfig":
    """Script 2 defaults (everything measured per PDF), exactly as on its command line."""
    ap = argparse.ArgumentParser()
    eng.add_engine_args(ap)
    return eng.engine_config_from_args(ap.parse_args([]))


def has_differences(summary: dict) -> bool:
    return any(summary.get(k, 0) for k in ("changed", "review", "entire_branches_removed", "entire_branches_added"))


def run_batch(pdf_a: str, pdf_b: str, out_dir: str, pairing: str = "auto", zoom: float = 2.0,
              mark_same: bool = False, log: Callable[[str], None] = print,
              progress: Optional[Callable[[int, int, str], None]] = None,
              cancel: Optional[threading.Event] = None) -> dict:
    """Run Script 2 once per page and Script 3 once per page pair; returns the batch summary."""
    for flag, path in (("PDF A", pdf_a), ("PDF B", pdf_b)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{flag} not found: {path}")
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    n_a, n_b = page_count(pdf_a), page_count(pdf_b)
    pairs, only_a, only_b = plan_pairs(n_a, n_b, pairing)
    os.makedirs(out_dir, exist_ok=True)
    seg_dir = os.path.join(out_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    log(f"PDF A: {pdf_a} ({n_a} page(s))")
    log(f"PDF B: {pdf_b} ({n_b} page(s))")
    log(f"Pairing: {pairing} -> {len(pairs)} page pair(s)")

    needed = sorted({("A", a) for a, _ in pairs} | {("B", b) for _, b in pairs})
    total = len(needed) + len(pairs)
    done = 0

    def step(text: str) -> None:
        nonlocal done
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(done, total, text)
        done += 1

    # ---- Script 2: segments of every page that takes part (each page once) ----------
    cfg = default_engine_config()
    seg_files: Dict[Tuple[str, int], str] = {}
    seg_errors: Dict[Tuple[str, int], str] = {}
    for tag, page in needed:
        pdf = pdf_a if tag == "A" else pdf_b
        step(f"Reading PDF {tag} page {page + 1}")
        t0 = time.time()
        try:
            data, _ = seg_builder.build_segments_for_pdf(pdf, page, cfg)
            n = len(data.get("segments") or [])
            if n == 0:
                raise RuntimeError("no harness wires found")
            path = os.path.join(seg_dir, f"{tag}_page{page + 1:02d}_segments.json")
            seg_builder.write_json(path, data)
            seg_files[(tag, page)] = path
            log(f"  PDF {tag} page {page + 1}: {n} segments ({time.time() - t0:.1f} s)")
        except Exception as exc:                                    # keep going with the other pages
            seg_errors[(tag, page)] = f"no harness drawing found on PDF {tag} page {page + 1}"
            log(f"  PDF {tag} page {page + 1}: skipped - no harness drawing found ({exc})")

    # ---- Script 3: every page pair -----------------------------------------------------
    rows: List[dict] = []
    for pa, pb in pairs:
        step(f"Comparing A page {pa + 1} with B page {pb + 1}")
        row = {"page_a": pa + 1, "page_b": pb + 1, "status": "", "message": "", "folder": "",
               "best_match": False, "summary": {}}
        missing = [seg_errors[k] for k in (("A", pa), ("B", pb)) if k in seg_errors]
        if missing:
            row.update(status="skipped", message="; ".join(missing))
            rows.append(row)
            continue
        folder = os.path.join(out_dir, pair_folder(pa, pb))
        row["folder"] = folder
        argv = ["--pdf-a", pdf_a, "--pdf-b", pdf_b,
                "--segments-a", seg_files[("A", pa)], "--segments-b", seg_files[("B", pb)],
                "--out", folder, "--page-a", str(pa), "--page-b", str(pb), "--zoom", str(zoom)]
        if mark_same:
            argv.append("--mark-same")
        buf = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = v6.main(argv)
        except Exception as exc:
            rc = 2
            buf.write(f"ERROR: {exc}\n")
        finally:
            v6._MS_CACHE.clear()                # caches are per comparison; keep memory flat
            v6._ALIGN_CACHE.clear()
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "log.txt"), "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())
        res_path = os.path.join(folder, "segment_comparison_v6.json")
        if rc != 0 or not os.path.isfile(res_path):
            err = [ln for ln in buf.getvalue().splitlines() if ln.startswith("ERROR")]
            row.update(status="error", message=err[-1] if err else f"comparison failed (code {rc})")
            log(f"  A p{pa + 1} vs B p{pb + 1}: {row['message']}")
            rows.append(row)
            continue
        with open(res_path, encoding="utf-8") as fh:
            sm = json.load(fh)["summary"]
        row["summary"] = {k: sm.get(k, 0) for k in SUMMARY_KEYS}
        row["status"] = "differences" if has_differences(sm) else "no differences"
        row["images"] = [os.path.join(folder, "pdf_a_segment_changes_v6.png"),
                         os.path.join(folder, "pdf_b_segment_changes_v6.png")]
        log(f"  A p{pa + 1} vs B p{pb + 1}: {row['status']}  (same {sm['same_exact'] + sm['same_probable']}, "
            f"changed {sm['changed']}, review {sm['review']}, removed {sm['entire_branches_removed']}, "
            f"added {sm['entire_branches_added']}; {time.time() - t0:.1f} s)")
        rows.append(row)

    # ---- best match when a page has several partners ---------------------------------
    def match_ratio(r: dict) -> float:
        s = r["summary"]
        return (s["same_exact"] + s["same_probable"]) / max(1, s["segments_a"], s["segments_b"])
    for key in ("page_a", "page_b"):
        groups: Dict[int, List[dict]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        for group in groups.values():
            done_rows = [r for r in group if r["summary"]]
            if len(group) > 1 and done_rows:
                max(done_rows, key=match_ratio)["best_match"] = True
    for r in rows:
        if r["summary"]:
            r["match_ratio"] = round(match_ratio(r), 3)
    for p in only_a:
        rows.append({"page_a": p + 1, "page_b": None, "status": "only in A", "best_match": False, "summary": {},
                     "folder": "", "message": "PDF B has no page with this number"})
    for p in only_b:
        rows.append({"page_a": None, "page_b": p + 1, "status": "only in B", "best_match": False, "summary": {},
                     "folder": "", "message": "PDF A has no page with this number"})
    if progress:
        progress(total, total, "Finished")

    result = {"pdf_a": os.path.abspath(pdf_a), "pdf_b": os.path.abspath(pdf_b), "pages_a": n_a, "pages_b": n_b,
              "pairing": pairing, "zoom": zoom, "pairs": rows,
              "totals": {"pairs": len(pairs),
                         "no_differences": sum(r["status"] == "no differences" for r in rows),
                         "with_differences": sum(r["status"] == "differences" for r in rows),
                         "skipped_or_failed": sum(r["status"] in ("skipped", "error") for r in rows),
                         "pages_only_in_a": len(only_a), "pages_only_in_b": len(only_b)}}
    write_summaries(result, out_dir)
    return result


def write_summaries(result: dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, "batch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    cols = ["page_a", "page_b", "status", "best_match", "match_ratio"] + list(SUMMARY_KEYS) + ["folder", "message"]
    with open(os.path.join(out_dir, "batch_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in result["pairs"]:
            w.writerow([r.get("page_a"), r.get("page_b"), r["status"], "yes" if r.get("best_match") else "",
                        r.get("match_ratio", "")] + [r["summary"].get(k, "") for k in SUMMARY_KEYS]
                       + [r.get("folder", ""), r.get("message", "")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare two harness PDFs page by page (single or multi-page).")
    ap.add_argument("--pdf-a", required=True, help="Baseline / old PDF.")
    ap.add_argument("--pdf-b", required=True, help="New / comparison PDF.")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--pairing", choices=PAIRINGS, default="auto", help="How pages are paired (default auto).")
    ap.add_argument("--zoom", type=float, default=2.0, help="Image zoom of the marked pages (default 2.0).")
    ap.add_argument("--mark-same", action="store_true", help="Also draw unchanged segments.")
    a = ap.parse_args(argv)
    try:
        res = run_batch(a.pdf_a, a.pdf_b, a.out, a.pairing, a.zoom, a.mark_same)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    t = res["totals"]
    print(f"Done: {t['pairs']} pair(s): {t['no_differences']} without differences, {t['with_differences']} with "
          f"differences, {t['skipped_or_failed']} skipped / failed; pages only in A {t['pages_only_in_a']}, "
          f"only in B {t['pages_only_in_b']}")
    print(f"Summary: {os.path.join(a.out, 'batch_summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
