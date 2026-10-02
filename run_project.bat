@echo off
echo ============================================
echo   AutoML-Lens - LLM-Powered AutoML Framework
echo ============================================
echo.

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Check Node
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

REM Install backend deps if needed
echo [1/4] Checking backend dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt -q

REM Create .env if missing
if not exist ".env" (
    echo [2/4] Creating default .env...
    copy "..\frontend\.env.example" ".env" >nul 2>&1
    echo LLM_PROVIDER=fallback> .env
    echo GEMINI_API_KEY=>> .env
    echo OPENAI_API_KEY=>> .env
    echo APP_NAME=AutoML-Lens>> .env
    echo STORAGE_PATH=storage>> .env
    echo DATABASE_URL=sqlite:///./automl_lens.db>> .env
    echo CORS_ORIGINS=http://localhost:5173,http://localhost:3000>> .env
)

REM Install frontend deps if needed
echo [3/4] Checking frontend dependencies...
cd /d "%~dp0frontend"
if not exist "node_modules" (
    npm install
)

REM Start both servers
echo [4/4] Starting servers...
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in either window to stop.
echo.

REM Start backend in new window
start "AutoML-Lens Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0"

REM Wait a moment then start frontend
timeout /t 3 /nobreak >nul
start "AutoML-Lens Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

REM Open browser
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo.
echo AutoML-Lens is starting. Check the opened windows.
pause
