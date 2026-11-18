@echo off
REM Checks that Laptop 2 can reach the PathSnap API on Laptop 1.
REM Usage: check_api.bat 192.168.43.25
setlocal
set "API_IP=%~1"
if "%API_IP%"=="" set /p "API_IP=Enter the IP address of Laptop 1: "

echo.
echo ===== 1. ping %API_IP% =====
ping -n 3 %API_IP%

echo.
echo ===== 2. GET http://%API_IP%:8000/health =====
curl -s -m 15 http://%API_IP%:8000/health
echo.

echo.
echo ===== 3. GET http://%API_IP%:8000/trips =====
curl -s -m 15 http://%API_IP%:8000/trips
echo.
echo.
echo If step 1 fails: the laptops are not on the same network (use a phone hotspot).
echo If step 1 works but 2 fails: firewall on Laptop 1 or the API is not running.
echo If 2 and 3 print JSON: everything is connected - run start_ui.bat.
pause
