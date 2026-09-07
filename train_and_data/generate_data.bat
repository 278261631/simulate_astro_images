@echo off
setlocal

REM Generate the (A, B) -> (dx, dy, droll) training dataset.
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   generate_data.bat --train 3000 --val 300 --test 300 --size 256
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

REM Pick a random seed each run so the content is never the same twice.
REM (seed is recorded in data\params.json; append --seed N to force a value)
set /a "SEED=(%RANDOM%*1000)+%RANDOM%"
echo Generating default dataset: 1500 train / 200 val / 200 test @ 192px  (seed %SEED%)
%PY% "%SCRIPT%" --train 100000 --val 200 --test 200 --seed %SEED% %*

echo.
echo Done. Data is in:
echo   %CD%\data
echo A few sample pairs were exported to:
echo   %CD%\check_data
if errorlevel 1 pause

endlocal
