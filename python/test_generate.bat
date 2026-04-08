@echo off
setlocal

REM Run from this script directory (python\).
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=render_sky_patch.py"
set "OUTDIR=test_outputs"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo [1/5] Default parameters (M42 center, FOV=30, Roll=0, soft PSF)
%PY% "%SCRIPT%" --out "%OUTDIR%\default.png"

echo [2/5] Narrow field
%PY% "%SCRIPT%" --fov 10 --out "%OUTDIR%\fov10.png"

echo [3/5] Camera roll clockwise 35 deg
%PY% "%SCRIPT%" --fov 10 --roll 35 --out "%OUTDIR%\fov10_roll35.png"

echo [4/5] Sharper stars
%PY% "%SCRIPT%" --fov 10 --psf-sigma 0.55 --gain 1.45 --out "%OUTDIR%\sharp.png"

echo [5/5] Softer glow
%PY% "%SCRIPT%" --fov 10 --psf-sigma 1.6 --gain 2.2 --max-mag 10.5 --out "%OUTDIR%\glow.png"

echo Done. Check output images in:
echo   %CD%\%OUTDIR%

endlocal
