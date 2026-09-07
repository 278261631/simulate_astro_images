@echo off
setlocal

REM Launch the PairRegNet validation GUI (PySide6).
REM Run from this script directory (train_and_data\).
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=validate_model_gui.py"

%PY% "%SCRIPT%"
if errorlevel 1 pause

endlocal
