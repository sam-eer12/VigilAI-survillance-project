@echo off
echo ========================================
echo VigilAI - Complete Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js first
    pause
    exit /b 1
)

echo ✓ Python and Node.js found

echo.
echo ========================================
echo Installing Backend Dependencies...
echo ========================================

REM Install Python packages
echo Installing Flask and AI dependencies...
pip install flask flask-cors ultralytics opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo WARNING: CUDA installation failed, trying CPU version...
    pip install torch torchvision torchaudio
)

echo ✓ Backend dependencies installed

echo.
echo ========================================
echo Installing Frontend Dependencies...
echo ========================================

REM Navigate to UI directory and install npm packages
cd vigilai-ui
echo Installing Next.js dependencies...
npm install

if errorlevel 1 (
    echo ERROR: Failed to install npm packages
    pause
    exit /b 1
)

echo ✓ Frontend dependencies installed

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start VigilAI:
echo.
echo 1. Start Backend (in main directory):
echo    python flask_api.py
echo.
echo 2. Start Frontend (in vigilai-ui directory):
echo    npm run dev
echo.
echo 3. Open your browser to:
echo    http://localhost:8080
echo.
echo Backend API runs on: http://localhost:5000
echo.
pause
