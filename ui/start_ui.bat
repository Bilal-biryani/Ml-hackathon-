@echo off
REM PathSnap evaluator UI launcher (Laptop 2)
REM Usage: start_ui.bat 192.168.43.25      (IP address of Laptop 1)
setlocal
cd /d "%~dp0"
set "API_IP=%~1"
if "%API_IP%"=="" set /p "API_IP=Enter the IP address of Laptop 1 (for example 192.168.43.25): "
if "%API_IP%"=="" (
  echo No IP entered. Exiting.
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Opening index.html directly instead.
  echo In the page, type http://%API_IP%:8000 into "API address" and click Connect.
  start "" "%~dp0index.html"
  pause
  exit /b 0
)

echo Starting the UI web server on port 5500 (a new window will open - keep it open)...
start "PathSnap UI server (keep open)" python -m http.server 5500
timeout /t 2 /nobreak >nul
start "" "http://localhost:5500/?api=http://%API_IP%:8000"
echo Browser opened at http://localhost:5500 connected to API http://%API_IP%:8000
pause
