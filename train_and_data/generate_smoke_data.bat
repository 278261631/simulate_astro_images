@echo off
setlocal

REM Generate ONLY the small independent smoke-test dataset as directly usable
REM PNG images + text labels (train\, val\, test\ sub-folders under
REM data_smoke\). Useful to quickly eyeball pairs or verify code changes
REM without touching the main numpy dataset in data\.
REM Default mix covers 64..512 px so the multi-scale model can be validated on
REM every resolution at once in validate_model_gui.py (see the "px" column and
REM the per-size summary lines).
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   generate_smoke_data.bat --smoke-sizes 192 --smoke-train 40
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

REM Pick a random seed each run so the smoke content is never the same twice.
REM (append --seed N to force a value and make it reproducible)
set /a "SEED=(%RANDOM%*1000)+%RANDOM%"
echo Generating smoke-test dataset (images + text): 20 train / 10 val / 10 test, sizes 64,128,192,384,512 mixed  (seed %SEED%)
%PY% "%SCRIPT%" --train 0 --val 0 --test 0 --smoke --seed %SEED% --smoke-train 20 --smoke-val 10 --smoke-test 10 --smoke-sizes 64,128,192,384,512 %*

echo.
echo Done. Smoke-test data (PNG pairs + TXT labels) is in:
echo   %CD%\data_smoke
if errorlevel 1 pause

endlocal
