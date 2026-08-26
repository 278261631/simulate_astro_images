@echo off
setlocal
cd /d "%~dp0"
python selftest.py
if errorlevel 1 (
    echo FAILED
    exit /b 1
)
echo PASSED
