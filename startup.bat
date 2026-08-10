@echo off
REM ============================================================================
REM Kairos - startup
REM
REM This script used to start Docker Desktop, bring up the QuantDinger compose
REM stack, and block on `curl http://localhost:8888/health` for 60 seconds
REM before launching the bot -- exiting with an error if that never answered.
REM QuantDinger has been removed; market data now comes straight from MT5. Left
REM as it was, this script could no longer start the bot at all: the health
REM check on a service that no longer exists always failed, and step 5 was
REM never reached.
REM
REM What remains is what the bot actually needs: MetaTrader 5 running, and a
REM populated .env.
REM ============================================================================

@echo off
title Kairos Startup
color 0A

echo ====================================
echo    Kairos - Startup
echo ====================================
echo.

cd /d "%~dp0"

REM ==============================
REM [1/4] Configuration
REM ==============================
echo [1/4] Checking configuration...
if not exist ".env" (
    echo.
    echo ERROR: .env not found in %CD%
    echo.
    echo Copy .env.example to .env and fill in MT5_LOGIN, MT5_PASSWORD
    echo and MT5_SERVER. The bot refuses to log in with an incomplete
    echo credential set, so a blank .env will stop it at startup.
    echo.
    pause
    exit /b 1
)
echo Found .env

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.10+ and retry.
    pause
    exit /b 1
)
echo Python is available

echo.
REM ==============================
REM [2/4] MetaTrader 5
REM ==============================
echo [2/4] Starting MetaTrader 5...
tasklist /FI "IMAGENAME eq terminal64.exe" 2>nul | find /I "terminal64.exe" >nul
if %errorlevel% equ 0 (
    echo MT5 is already running.
    goto MT5_READY
)

if not exist "C:\Program Files\MetaTrader 5\terminal64.exe" (
    echo ERROR: MT5 not found at C:\Program Files\MetaTrader 5\terminal64.exe
    echo If it is installed elsewhere, set MT5_PATH in .env and start it manually.
    pause
    exit /b 1
)

start "" "C:\Program Files\MetaTrader 5\terminal64.exe"
echo Waiting for MT5 to start...

REM Bounded wait. The old script looped forever on Docker with no way out.
set MT5_TRIES=0
:WAIT_MT5
timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq terminal64.exe" 2>nul | find /I "terminal64.exe" >nul
if %errorlevel% equ 0 goto MT5_READY
set /a MT5_TRIES+=1
echo Still waiting for MT5... attempt %MT5_TRIES%
if %MT5_TRIES% lss 24 goto WAIT_MT5
echo.
echo ERROR: MT5 did not start within 2 minutes.
pause
exit /b 1

:MT5_READY
echo MT5 is running.
REM The terminal needs a moment after the process appears before its Python
REM API accepts a connection.
echo Giving the terminal time to finish loading...
timeout /t 15 /nobreak > nul

echo.
REM ==============================
REM [3/4] Connection check
REM ==============================
echo [3/4] Verifying the MT5 connection...
python -c "from data.market.mt5_session import ensure_session, get_account_info; import sys; ok = ensure_session(); acc = get_account_info() if ok else None; print(f'Connected: login={acc.login} balance={acc.balance}') if acc else print('Connection failed'); sys.exit(0 if acc else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: could not establish an MT5 session.
    echo Check MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env, and that the
    echo terminal is logged in and allows algorithmic trading.
    echo.
    pause
    exit /b 1
)

echo.
REM ==============================
REM [4/4] Bot
REM ==============================
echo [4/4] Starting Kairos...
start "Kairos" cmd /k "python main.py"

echo.
echo ====================================
echo    Started. Check Telegram.
echo ====================================
pause
