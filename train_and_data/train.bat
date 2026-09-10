@echo off
setlocal

REM Train the pose + transient-detection CNN on the tr dataset.
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   train.bat --epochs 60 --batch 32 --det-w 0.3
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=train.py"

echo Training PairRegNet + transient head (45 epochs by default, det-w 0.5)
%PY% "%SCRIPT%" --data data_tr --model-dir models_tr --epochs 45 --det-w 0.5 %*

echo.
echo Done. Checkpoint saved in:
echo   %CD%\models_tr\best.pt
if errorlevel 1 pause

endlocal
