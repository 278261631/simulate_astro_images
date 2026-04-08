@echo off
setlocal

cd /d "%~dp0"

set "GPP=C:\msys64\mingw64\bin\g++.exe"
set "EXE=render_sky_patch.exe"
set "SRC=render_sky_patch.cpp"
set "OUTDIR=test_outputs"

REM Ensure MSYS2 runtime DLLs take precedence over other toolchains in PATH.
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"

if not exist "%GPP%" (
  echo g++ not found: %GPP%
  exit /b 1
)

echo Building %SRC% ...
"%GPP%" -std=c++17 -O2 -o "%EXE%" "%SRC%"
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running default render ...
"%EXE%" --out "%OUTDIR%\default.bmp"
if errorlevel 1 (
  echo Run failed.
  exit /b 1
)

echo Done.
echo Output folder: %CD%\%OUTDIR%

endlocal
