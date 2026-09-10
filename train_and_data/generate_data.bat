@echo off
setlocal

REM Generate the transient (A, B) -> (dx, dy, droll) training dataset.
REM Run from this script directory (train_and_data\).
REM Defaults target the tr pipeline: 256px frames with simulated transients
REM (new / brighten / move) and GT in data_tr\*_meta.npz.
REM Extra arguments are appended, e.g.:
REM   generate_data.bat --train 5000 --transient-rate 0
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

REM Pick a random seed each run so the content is never the same twice.
REM (seed is recorded in data_tr\params.json; append --seed N to force a value)
set /a "SEED=(%RANDOM%*1000)+%RANDOM%"
echo Generating transient dataset: 20000 train / 200 val / 200 test @ 256px  (seed %SEED%)
%PY% "%SCRIPT%" --out data_tr --size 256 --train 20000 --val 200 --test 200 --seed %SEED% %*

echo.
echo Done. Data is in:
echo   %CD%\data_tr
echo A few sample pairs were exported to:
echo   %CD%\check_data
if errorlevel 1 pause

endlocal
