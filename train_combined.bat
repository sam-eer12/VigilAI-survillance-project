@echo off
echo ================================
echo VigilAI Combined Model Training
echo ================================
echo.
echo This will train a YOLOv11 medium model using all three datasets:
echo 1. HackByte_Dataset (objects)
echo 2. FireSmokeNEWdataset (fire/smoke)  
echo 3. violence-detection-dataset (actions)
echo.
echo Training will be performed on CPU for compatibility.
echo This process may take several hours.
echo.
pause

echo Installing/updating required packages...
pip install -r requirements_combined.txt

echo.
echo Starting combined model training...
python run_combined_training.py

echo.
echo Training completed! Check the output above for results.
pause
