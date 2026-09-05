@echo off
rem Double-clickable entry point. Runs the PowerShell bootstrap with a
rem per-process execution-policy bypass so nothing machine-wide changes,
rem then pauses so the window does not vanish before an error can be read.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
echo.
pause
