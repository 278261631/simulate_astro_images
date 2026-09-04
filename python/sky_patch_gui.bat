@echo off
setlocal

REM Launcher for the PySide6 sky-patch GUI.
REM Run from this script directory (python\).
cd /d "%~dp0"

set "PY=python"
set "SCRIPT=sky_patch_gui.py"

%PY% "%SCRIPT%"
if errorlevel 1 pause

endlocal
