@echo off
setlocal

REM Generate a high-resolution (512x512) (A, B) -> (dx, dy, droll) dataset.
REM Use together with train.bat --roi-min 96 --roi-max 512 so the model is
REM trained on random-resolution crops that cover ~64px..512px content scale
REM (the network input is fixed at 192x192; labels are rescaled accordingly).
REM Run from this script directory (train_and_data\).
REM Extra arguments are appended, e.g.:
REM   generate_data_512.bat --train 10000 --val 200 --test 200
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=generate_dataset.py"

REM PSF sigma kept wider than the 192px default: after ROI down-sampling the
REM stars must still show real (not purely interpolated) structure across the
REM whole crop-size range.
set /a "SEED=(%RANDOM%*1000)+%RANDOM%"
echo Generating high-res dataset: 20000 train / 200 val / 200 test @ 512px  (seed %SEED%)
%PY% "%SCRIPT%" --size 512 --train 20000 --val 200 --test 200 --seed %SEED% --psf-sigma-min 0.6 --psf-sigma-max 2.4 %*

echo.
echo Done. Data is in:
echo   %CD%\data
echo A few sample pairs were exported to:
echo   %CD%\check_data
if errorlevel 1 pause

endlocal
