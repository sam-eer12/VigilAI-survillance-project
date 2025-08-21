@echo off
echo Setting up VigilAI Environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo Python found. Installing dependencies...

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (change to cpu version if no GPU)
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo Usage examples:
echo 1. Extract dataset: python vigilai.py --mode extract
echo 2. Train model: python vigilai.py --mode train --epochs 50
echo 3. Live detection: python vigilai.py --mode detect --model path/to/model.pt
echo 4. Evaluate model: python vigilai.py --mode evaluate --model path/to/model.pt
echo.
pause
