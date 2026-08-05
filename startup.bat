@echo off
title Trading Bot V3 Startup
color 0A

echo ====================================
echo    Trading Bot V3 - Auto Startup
echo ====================================
echo.

:: ==============================
:: [1/5] Docker
:: ==============================
echo [1/5] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to be ready...
    :WAIT_DOCKER
    timeout /t 5 /nobreak > nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 goto WAIT_DOCKER
    echo Docker is ready!
) else (
    echo Docker is already running.
)

echo.
:: ==============================
:: [2/5] QuantDinger Containers
:: ==============================
echo [2/5] Starting QuantDinger containers...
cd /d "C:\Users\ACER\QuantDinger\QuantDinger"
if %errorlevel% neq 0 (
    echo ERROR: QuantDinger path not found!
    pause
    exit /b 1
)

docker compose up -d
timeout /t 5 /nobreak > nul
docker compose stop backend
timeout /t 3 /nobreak > nul
echo QuantDinger containers ready!

echo.
:: ==============================
:: [3/5] MT5
:: ==============================
echo [3/5] Starting MT5...
start "" "C:\Program Files\MetaTrader 5\terminal64.exe"
echo Waiting for MT5...
:WAIT_MT5
timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq terminal64.exe" 2>nul | find /I "terminal64.exe" >nul
if %errorlevel% neq 0 goto WAIT_MT5
echo MT5 is running!
timeout /t 10 /nobreak > nul

echo.
:: ==============================
:: [4/5] QuantDinger Backend
:: ==============================
echo [4/5] Starting QuantDinger Backend...
cd /d "C:\Users\ACER\QuantDinger\QuantDinger\backend_api_python"
if %errorlevel% neq 0 (
    echo ERROR: Backend path not found!
    pause
    exit /b 1
)

start "QD Backend" cmd /k "python run.py"

echo Waiting for backend to be ready...
set RETRY=0
:WAIT_BACKEND
timeout /t 3 /nobreak > nul
curl -s http://localhost:8888/health >nul 2>&1
if %errorlevel% neq 0 (
    set /a RETRY+=1
    echo Still waiting... attempt %RETRY%
    if %RETRY% lss 20 goto WAIT_BACKEND
    echo.
    echo ERROR: Backend did not start after 60s!
    echo Check QD Backend window for errors.
    pause
    exit /b 1
)
echo Backend is ready!

echo.
:: ==============================
:: [5/5] Trading Bot
:: ==============================
echo [5/5] Starting Trading Bot V3...
cd /d "%~dp0"
start "Bot V3" cmd /k "python main.py"

echo.
echo ====================================
echo    V3 Started! Check Telegram
echo ====================================
pause
