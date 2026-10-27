#!/bin/bash
# Build dist/HarnessCompare (macOS / Linux desktop application). Needs Python 3.10+ with tkinter
# (macOS: python.org installer; Linux: e.g. "sudo apt install python3-tk python3-venv").
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
# --collect-all pymupdf : PyMuPDF needs all of its compiled parts (text positions depend on them)
# --exclude-module ...   : optional page-layout AI model, not used here (smaller, faster start)
pyinstaller --noconfirm --clean --onefile --windowed --name HarnessCompare \
    --collect-all pymupdf --hidden-import batch_compare \
    --exclude-module pymupdf.layout --exclude-module onnxruntime \
    harness_compare_app.py
echo "Built: $(pwd)/dist/HarnessCompare"
