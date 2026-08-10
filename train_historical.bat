@echo off
REM ============================================================================
REM Kairos - entry model training
REM
REM This menu used to drive train_from_historical.py, which could not run: it
REM imported a module that does not exist in the repository, so every option
REM died at import. Training now lives in scripts/, split into a fetch step
REM (needs MT5, so Windows only) and a train step.
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ========================================
echo   Kairos - Entry Model Training
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    pause
    exit /b 1
)

echo Select an option:
echo.
echo   1. Fetch historical candles from MT5   (needs MT5 running + logged in)
echo   2. Validate only - build dataset, walk-forward, no model written
echo   3. Train and install the model         (backs up the current one first)
echo   4. Full run - fetch, then train and install
echo.

set /p choice="Enter your choice (1-4, or press Enter for 4): "
if "%choice%"=="" set choice=4

if "%choice%"=="1" (
    python scripts\fetch_training_candles.py
) else if "%choice%"=="2" (
    python scripts\train_entry_model.py --dry-run
) else if "%choice%"=="3" (
    python scripts\train_entry_model.py
) else if "%choice%"=="4" (
    python scripts\fetch_training_candles.py
    if errorlevel 1 (
        echo.
        echo Fetch failed - not training on stale or missing candles.
        pause
        exit /b 1
    )
    python scripts\train_entry_model.py
) else (
    echo Invalid choice
    pause
    exit /b 1
)

set RESULT=%errorlevel%

echo.
if %RESULT% neq 0 (
    echo ========================================
    echo   FAILED - see the messages above.
    echo   The live model was NOT replaced.
    echo ========================================
) else (
    echo ========================================
    echo   Done.
    echo   Candles: data\historical\
    echo   Model:   models\entry\entry_model.json
    echo   Report:  models\entry\training_report.json
    echo ========================================
)
pause
exit /b %RESULT%
