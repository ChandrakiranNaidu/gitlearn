@echo off
rem Build dist\HarnessCompare.exe (Windows desktop application).
rem Needs 64-bit Python 3.10.1 or newer from python.org (3.12 recommended; includes tkinter).
rem Python 3.10.0 cannot be used: a bug in it breaks PyInstaller ("IndexError: tuple index out of range").
rem Double-click this file, or run it from a command prompt in this folder.
setlocal
cd /d "%~dp0"

rem ---- pick a Python: the newest one known to the "py" launcher, else "python" on PATH
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY set "PY=python"
%PY% -c "import sys; sys.exit(0 if sys.version_info[:3] >= (3, 10, 1) else 1)" >nul 2>&1 || goto :badpython
%PY% -c "import tkinter" >nul 2>&1 || goto :notk
echo Using:
%PY% --version

rem ---- private environment; one made earlier by an unusable Python (e.g. 3.10.0) is rebuilt
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import sys, tkinter; sys.exit(0 if sys.version_info[:3] >= (3, 10, 1) else 1)" >nul 2>&1 || rmdir /s /q .venv
)
if not exist .venv\Scripts\python.exe (
    %PY% -m venv .venv || goto :error
)
set "VPY=.venv\Scripts\python.exe"
%VPY% -m pip install --upgrade pip || goto :error
%VPY% -m pip install -r requirements.txt pyinstaller || goto :error
if exist build rmdir /s /q build

rem --collect-all pymupdf : PyMuPDF needs all of its compiled parts (text positions depend on them)
rem --exclude-module ...   : optional page-layout AI model, not used here (smaller, faster start)
%VPY% -m PyInstaller --noconfirm --clean --onefile --windowed --name HarnessCompare ^
    --collect-all pymupdf --hidden-import batch_compare ^
    --exclude-module pymupdf.layout --exclude-module onnxruntime ^
    harness_compare_app.py || goto :error

echo.
echo Built: %~dp0dist\HarnessCompare.exe
pause
exit /b 0

:badpython
echo.
echo This Python cannot build the exe:
%PY% --version
echo Python 3.10.0 has a bug that breaks PyInstaller - "IndexError: tuple index out of range".
echo Install Python 3.12 64-bit from https://www.python.org/downloads/windows/
echo No administrator rights are needed. Tick "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:notk
echo.
echo This Python has no tkinter, which the application window needs.
echo Reinstall Python from python.org with the "tcl/tk and IDLE" option ticked, then run this file again.
pause
exit /b 1

:error
echo.
echo Build failed - see the messages above.
pause
exit /b 1
