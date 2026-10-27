#!/bin/bash
# Build dist/HarnessCompare (macOS / Linux desktop application). Needs Python 3.10+ with tkinter
# (macOS: python.org installer; Linux: e.g. "sudo apt install python3-tk python3-venv").
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
OK='import sys, tkinter; sys.exit(0 if sys.version_info[:3] >= (3, 10, 1) else 1)'
if ! "$PY" -c "$OK" 2>/dev/null; then
    echo "$("$PY" --version 2>&1) cannot build the app: Python 3.10.1 or newer with tkinter is needed"
    echo "(Python 3.10.0 has a bug that breaks PyInstaller: IndexError: tuple index out of range)."
    exit 1
fi
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "$OK" 2>/dev/null; then
    rm -rf .venv                        # made earlier by an unusable Python
fi
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
