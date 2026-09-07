@echo off
setlocal

REM Generate ONLY the small independent smoke-test dataset as directly usable
REM PNG images + text labels (train\, val\, test\ sub-folders under
REM data_smoke\). Useful to quickly eyeball pairs or verify code changes
REM without touching the main numpy dataset in data\.
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   generate_smoke_data.bat --smoke-train 40 --smoke-val 10 --smoke-test 10 --smoke-size 128
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

echo Generating smoke-test dataset (images + text): 16 train / 8 val / 8 test @ 96px
%PY% "%SCRIPT%" --train 0 --val 0 --test 0 --smoke %*

echo.
echo Done. Smoke-test data (PNG pairs + TXT labels) is in:
echo   %CD%\data_smoke
if errorlevel 1 pause

endlocal
