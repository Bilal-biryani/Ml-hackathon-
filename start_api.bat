@echo off
REM PathSnap API launcher (Laptop 1)
REM   start_api.bat                      serve the evaluator trips in data\eval\
REM   start_api.bat "D:\path\to\trips"   serve trip folders from another folder
REM   start_api.bat rehearsal            serve the 10 student-test trips (practice run)
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 and run:  pip install -r requirements.txt
  pause
  exit /b 1
)
python -c "import fastapi, uvicorn, lightgbm, sklearn" 2>nul
if errorlevel 1 (
  echo Missing packages. Run:  pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist "models\final_model.joblib" (
  echo models\final_model.joblib is missing. Pull the latest repository first.
  pause
  exit /b 1
)

if /i "%~1"=="rehearsal" (
  set "PATHSNAP_TRIPS_DIR=%~dp0data\raw\student_test"
) else if not "%~1"=="" (
  set "PATHSNAP_TRIPS_DIR=%~1"
)
if "%PATHSNAP_TRIPS_DIR%"=="" set "PATHSNAP_TRIPS_DIR=%~dp0data\eval"

echo.
echo ================= PathSnap API (Laptop 1) =================
echo Trips folder: %PATHSNAP_TRIPS_DIR%
echo.
echo IP addresses of this laptop (type one of these on Laptop 2 as http://IP:8000):
ipconfig | findstr /R /C:"IPv4"
echo.
echo If Windows asks to allow Python through the firewall, tick BOTH Private and Public and click Allow.
echo Keep this window open: every request from the UI is logged here (show these logs to the evaluator).
echo ============================================================
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-use-colors
pause
