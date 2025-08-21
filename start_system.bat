@echo off
echo Starting VigilAI Surveillance System...
echo.

echo Starting Flask Backend...
start "VigilAI Backend" cmd /k "python flask_api.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting React Frontend...
start "VigilAI Frontend" cmd /k "cd vigilai-ui && npm run dev"

echo.
echo VigilAI System is starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause > nul
