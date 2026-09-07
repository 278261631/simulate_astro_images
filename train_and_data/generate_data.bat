@echo off
setlocal

REM Generate the (A, B) -> (dx, dy, droll) training dataset.
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   generate_data.bat --train 3000 --val 300 --test 300 --size 256
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

echo Generating default dataset: 1500 train / 200 val / 200 test @ 192px
%PY% "%SCRIPT%" --train 1500 --val 200 --test 200 %*

echo.
echo Done. Data is in:
echo   %CD%\data
echo A few sample pairs were exported to:
echo   %CD%\check_data
if errorlevel 1 pause

endlocal
