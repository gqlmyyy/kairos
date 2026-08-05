@echo off
REM Trading Bot V3 - train_historical.bat
REM Quick launcher for historical data training

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Historical Data Training Pipeline
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Show menu
echo Select an option:
echo.
echo 1. Fetch data only (save CSV files)
echo 2. Build dataset only (process existing CSV)
echo 3. Train model only (use existing database)
echo 4. Full pipeline (fetch + build + train) - DEFAULT
echo 5. Show database statistics
echo 6. Fetch 2 years of data (longer history)
echo 7. Custom symbols only
echo.

set /p choice="Enter your choice (1-7, or press Enter for 4): "

if "%choice%"=="" set choice=4

if "%choice%"=="1" (
    echo Running: Fetch only...
    python train_from_historical.py --fetch-only
) else if "%choice%"=="2" (
    echo Running: Build dataset...
    python train_from_historical.py --train-only --fetch-only
) else if "%choice%"=="3" (
    echo Running: Train model...
    python train_from_historical.py --train-only
) else if "%choice%"=="4" (
    echo Running: Full pipeline...
    python train_from_historical.py
) else if "%choice%"=="5" (
    echo Running: Database stats...
    python train_from_historical.py --stats
) else if "%choice%"=="6" (
    echo Running: Fetch 2 years...
    python train_from_historical.py --days 730
) else if "%choice%"=="7" (
    set /p symbols="Enter symbols separated by space (e.g. EURUSD XAUUSD): "
    python train_from_historical.py --symbols !symbols!
) else (
    echo Invalid choice
    pause
    exit /b 1
)

echo.
echo ========================================
echo Process completed!
echo.
echo Training data location: data/historical/
echo Training samples location: data/training/
echo Model location: models/
echo.
pause
