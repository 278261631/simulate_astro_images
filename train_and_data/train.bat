@echo off
setlocal

REM Train the pair-regression CNN on the generated dataset.
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   train.bat --epochs 80 --batch 64 --data data
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=train.py"

echo Training PairRegNet (45 epochs by default)
%PY% "%SCRIPT%" --epochs 45 %*

echo.
echo Done. Checkpoint saved in:
echo   %CD%\models\best.pt
if errorlevel 1 pause

endlocal
