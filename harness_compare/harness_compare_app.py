#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harness PDF Compare - desktop application.

Pick the old and the new harness PDF (single-page or multi-page), choose how the pages
are paired and press Compare. Every page pair goes through the structural pipeline
(Scripts 1-3). The results table shows each pair; double-click a row to open its marked
images. Everything is written to the output folder (see batch_compare.py).

    python harness_compare_app.py                                   open the window
    python harness_compare_app.py --pdf-a A.pdf --pdf-b B.pdf --out results [--pairing ...]
                                                                    run without a window
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback

PAIRING_LABELS = {
    "Automatic: one page vs every page, otherwise page by page": "auto",
    "Page by page (1-1, 2-2, 3-3, ...)": "page-by-page",
    "Every page against every page (page order may differ)": "all-pairs",
}
RESULT_TEXT = {"no differences": "No differences", "differences": "Differences found",
               "skipped": "Skipped", "error": "Error", "only in A": "Page only in A", "only in B": "Page only in B"}
ROW_TAG = {"no differences": "same", "differences": "diff", "skipped": "skip", "error": "error",
           "only in A": "skip", "only in B": "skip"}


def open_path(path: str) -> None:
    """Open a file or folder with the system's default application."""
    if not path or not os.path.exists(path):
        return
    if sys.platform.startswith("win"):
        os.startfile(path)                                          # noqa: S606 (Windows only)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def count_pages(path: str) -> int:
    import pymupdf
    with pymupdf.open(path) as doc:
        return doc.page_count


class App:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("Harness PDF Compare")
        root.minsize(900, 620)
        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self.rows = {}
        self.out_dir = ""

        self.var_a, self.var_b, self.var_out = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.info_a, self.info_b = tk.StringVar(), tk.StringVar()
        self.var_pairing = tk.StringVar(value=next(iter(PAIRING_LABELS)))
        self.var_zoom = tk.StringVar(value="2.0")
        self.var_mark_same = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Choose the two PDFs to compare.")

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        def file_row(r, label, var, info, command):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(frm, textvariable=var).grid(row=r, column=1, sticky="ew", padx=6)
            b = ttk.Button(frm, text="Browse...", command=command)
            b.grid(row=r, column=2, sticky="ew")
            if info is not None:
                ttk.Label(frm, textvariable=info, width=14).grid(row=r, column=3, sticky="w", padx=6)
            return b
        self.btn_a = file_row(0, "Old PDF (A)", self.var_a, self.info_a, lambda: self.pick_pdf("A"))
        self.btn_b = file_row(1, "New PDF (B)", self.var_b, self.info_b, lambda: self.pick_pdf("B"))
        self.btn_out = file_row(2, "Output folder", self.var_out, None, self.pick_out)

        opts = ttk.Frame(frm)
        opts.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        ttk.Label(opts, text="Page pairing").pack(side="left")
        ttk.Combobox(opts, textvariable=self.var_pairing, values=list(PAIRING_LABELS), state="readonly",
                     width=46).pack(side="left", padx=6)
        ttk.Label(opts, text="Image zoom").pack(side="left", padx=(12, 0))
        ttk.Spinbox(opts, textvariable=self.var_zoom, from_=1.0, to=4.0, increment=0.5, width=5).pack(side="left", padx=6)
        ttk.Checkbutton(opts, text="Also draw unchanged segments", variable=self.var_mark_same).pack(side="left", padx=12)

        run = ttk.Frame(frm)
        run.grid(row=4, column=0, columnspan=4, sticky="ew", pady=4)
        run.columnconfigure(2, weight=1)
        self.btn_run = ttk.Button(run, text="Compare", command=self.start)
        self.btn_run.grid(row=0, column=0)
        self.btn_cancel = ttk.Button(run, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=1, padx=6)
        self.progress = ttk.Progressbar(run, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Label(frm, textvariable=self.var_status).grid(row=5, column=0, columnspan=4, sticky="w")

        cols = ("page_a", "page_b", "result", "same", "changed", "review", "removed", "added", "best")
        heads = ("Page A", "Page B", "Result", "Same", "Changed", "Review", "Removed", "Added", "Best match")
        widths = (70, 70, 150, 70, 70, 70, 75, 70, 85)
        tree_frame = ttk.Frame(frm)
        tree_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=6)
        frm.rowconfigure(6, weight=3)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w" if c == "result" else "center", stretch=(c == "result"))
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.tag_configure("same", background="#dff3df")
        self.tree.tag_configure("diff", background="#ffe2bf")
        self.tree.tag_configure("skip", background="#eeeeee")
        self.tree.tag_configure("error", background="#f6d0d0")
        self.tree.bind("<Double-1>", lambda e: self.open_selected("images"))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.update_buttons())

        acts = ttk.Frame(frm)
        acts.grid(row=7, column=0, columnspan=4, sticky="ew")
        self.btn_img_a = ttk.Button(acts, text="Open marked page A", command=lambda: self.open_selected("a"))
        self.btn_img_b = ttk.Button(acts, text="Open marked page B", command=lambda: self.open_selected("b"))
        self.btn_pair = ttk.Button(acts, text="Open pair folder", command=lambda: self.open_selected("folder"))
        self.btn_csv = ttk.Button(acts, text="Open summary (CSV)",
                                  command=lambda: open_path(os.path.join(self.out_dir, "batch_summary.csv")))
        self.btn_outdir = ttk.Button(acts, text="Open output folder", command=lambda: open_path(self.out_dir))
        for b in (self.btn_img_a, self.btn_img_b, self.btn_pair, self.btn_csv, self.btn_outdir):
            b.pack(side="left", padx=(0, 6))
            b.state(["disabled"])

        ttk.Label(frm, text="Log").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.log_box = ScrolledText(frm, height=9, wrap="word", state="disabled", font=("Courier New", 9))
        self.log_box.grid(row=9, column=0, columnspan=4, sticky="nsew")
        frm.rowconfigure(9, weight=2)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)

    # ---- inputs ----------------------------------------------------------------------
    def pick_pdf(self, tag: str) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(title=f"Choose PDF {tag}", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not path:
            return
        (self.var_a if tag == "A" else self.var_b).set(path)
        info = self.info_a if tag == "A" else self.info_b
        try:
            n = count_pages(path)
            info.set(f"{n} page{'s' if n != 1 else ''}")
        except Exception as exc:
            info.set("not a PDF")
            self.log(f"Cannot open {path}: {exc}")
        if not self.var_out.get() and self.var_a.get() and self.var_b.get():
            a = os.path.splitext(os.path.basename(self.var_a.get()))[0]
            b = os.path.splitext(os.path.basename(self.var_b.get()))[0]
            self.var_out.set(os.path.join(os.path.dirname(self.var_a.get()), f"{a}_vs_{b}_comparison"))

    def pick_out(self) -> None:
        from tkinter import filedialog
        path = filedialog.askdirectory(title="Choose the output folder")
        if path:
            self.var_out.set(path)

    # ---- running -----------------------------------------------------------------------
    def start(self) -> None:
        from tkinter import messagebox
        a, b, out = self.var_a.get().strip(), self.var_b.get().strip(), self.var_out.get().strip()
        problems = [m for ok, m in ((os.path.isfile(a), "Choose the old PDF (A)."),
                                    (os.path.isfile(b), "Choose the new PDF (B)."),
                                    (bool(out), "Choose an output folder.")) if not ok]
        try:
            zoom = float(self.var_zoom.get())
            if not 0.5 <= zoom <= 6:
                raise ValueError
        except ValueError:
            problems.append("Image zoom must be a number between 0.5 and 6.")
            zoom = 2.0
        if problems:
            messagebox.showwarning("Harness PDF Compare", "\n".join(problems))
            return
        self.out_dir = out
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.cancel_event.clear()
        self.set_running(True)
        pairing = PAIRING_LABELS[self.var_pairing.get()]
        self.worker = threading.Thread(target=self.work, args=(a, b, out, pairing, zoom, self.var_mark_same.get()),
                                       daemon=True)
        self.worker.start()

    def work(self, a, b, out, pairing, zoom, mark_same) -> None:
        try:
            import batch_compare
            res = batch_compare.run_batch(
                a, b, out, pairing, zoom, mark_same,
                log=lambda s: self.queue.put(("log", s)),
                progress=lambda d, t, txt: self.queue.put(("progress", d, t, txt)),
                cancel=self.cancel_event)
            self.queue.put(("done", res))
        except Exception as exc:
            if type(exc).__name__ == "Cancelled":
                self.queue.put(("cancelled",))
            else:
                self.queue.put(("error", f"{exc}", traceback.format_exc()))

    def cancel(self) -> None:
        self.cancel_event.set()
        self.var_status.set("Cancelling after the current step...")

    def poll(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.log(msg[1])
                elif kind == "progress":
                    _, d, t, txt = msg
                    self.progress.configure(maximum=max(1, t), value=d)
                    self.var_status.set(f"{txt}  ({d}/{t})")
                elif kind == "done":
                    self.show_results(msg[1])
                    self.set_running(False)
                elif kind == "cancelled":
                    self.var_status.set("Cancelled.")
                    self.set_running(False)
                elif kind == "error":
                    from tkinter import messagebox
                    self.log(msg[2])
                    self.var_status.set("Failed: " + msg[1])
                    self.set_running(False)
                    messagebox.showerror("Harness PDF Compare", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def set_running(self, running: bool) -> None:
        for b in (self.btn_run, self.btn_a, self.btn_b, self.btn_out):
            b.state(["disabled"] if running else ["!disabled"])
        self.btn_cancel.state(["!disabled"] if running else ["disabled"])
        self.update_buttons()

    # ---- results -----------------------------------------------------------------------
    def show_results(self, res: dict) -> None:
        for r in res["pairs"]:
            s = r.get("summary") or {}
            text = RESULT_TEXT.get(r["status"], r["status"])
            if r["status"] in ("skipped", "error") and r.get("message"):
                text += ": " + r["message"]
            vals = (r.get("page_a") or "-", r.get("page_b") or "-", text,
                    (s["same_exact"] + s["same_probable"]) if s else "", s.get("changed", ""), s.get("review", ""),
                    s.get("entire_branches_removed", ""), s.get("entire_branches_added", ""),
                    "yes" if r.get("best_match") else "")
            iid = self.tree.insert("", "end", values=vals, tags=(ROW_TAG.get(r["status"], "skip"),))
            self.rows[iid] = r
        t = res["totals"]
        self.var_status.set(f"Finished: {t['pairs']} page pair(s) - {t['no_differences']} without differences, "
                            f"{t['with_differences']} with differences, {t['skipped_or_failed']} skipped / failed. "
                            f"Double-click a row to open its marked pages.")
        self.progress.configure(value=self.progress["maximum"])

    def selected(self):
        sel = self.tree.selection()
        return self.rows.get(sel[0]) if sel else None

    def open_selected(self, what: str) -> None:
        r = self.selected()
        if not r or not r.get("folder"):
            return
        imgs = r.get("images") or []
        if what == "folder":
            open_path(r["folder"])
        elif what in ("a", "b") and imgs:
            open_path(imgs[0 if what == "a" else 1])
        elif what == "images" and imgs:
            for p in imgs:
                open_path(p)
        else:
            open_path(r["folder"])

    def update_buttons(self) -> None:
        running = self.worker is not None and self.worker.is_alive()
        r = self.selected()
        has_pair = bool(r and r.get("images")) and not running
        for b in (self.btn_img_a, self.btn_img_b, self.btn_pair):
            b.state(["!disabled"] if has_pair else ["disabled"])
        has_out = bool(self.out_dir) and os.path.isdir(self.out_dir) and not running
        self.btn_outdir.state(["!disabled"] if has_out else ["disabled"])
        has_csv = has_out and os.path.isfile(os.path.join(self.out_dir, "batch_summary.csv"))
        self.btn_csv.state(["!disabled"] if has_csv else ["disabled"])

    def log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def on_close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:                       # command-line use: no window
        import batch_compare
        return batch_compare.main(sys.argv[1:])
    if sys.platform.startswith("win"):
        try:                                    # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
