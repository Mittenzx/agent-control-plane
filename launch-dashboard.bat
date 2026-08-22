@echo off
title Agent Control Plane Dashboard
echo ============================================
echo   Agent Control Plane Dashboard
echo   Starting server on http://localhost:8090
echo   Press Ctrl+C to stop
echo ============================================
echo.

cd /d "%~dp0"

REM Open the browser after a short delay
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8090"

REM Run the dashboard server
python -m uvicorn control_plane.web.server:app --host 127.0.0.1 --port 8090

echo.
echo Server stopped.
pause
