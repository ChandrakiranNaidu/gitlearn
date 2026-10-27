@echo off
rem Build dist\HarnessCompare.exe (Windows desktop application).
rem Needs 64-bit Python 3.10 or newer from python.org (includes tkinter).
rem Double-click this file, or run it from a command prompt in this folder.

cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt pyinstaller || goto :error

rem --collect-all pymupdf : PyMuPDF needs all of its compiled parts (text positions depend on them)
rem --exclude-module ...   : optional page-layout AI model, not used here (smaller, faster start)
pyinstaller --noconfirm --clean --onefile --windowed --name HarnessCompare ^
    --collect-all pymupdf --hidden-import batch_compare ^
    --exclude-module pymupdf.layout --exclude-module onnxruntime ^
    harness_compare_app.py || goto :error

echo.
echo Built: %~dp0dist\HarnessCompare.exe
pause
exit /b 0

:error
echo.
echo Build failed - see the messages above.
pause
exit /b 1
