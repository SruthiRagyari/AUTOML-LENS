@echo off
echo ============================================
echo   AutoML-Lens: Starting Application
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)

:: Create .env if not exists
if not exist ".env" (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env
    echo [INFO] Edit .env to add your API keys (optional - fallback mode works without them)
)

:: Install backend dependencies
echo.
echo [1/4] Installing backend dependencies...
cd backend
pip install -r requirements.txt --quiet
cd ..

:: Install frontend dependencies
echo.
echo [2/4] Installing frontend dependencies...
cd frontend
call npm install --silent
cd ..

:: Start backend
echo.
echo [3/4] Starting backend server...
start "AutoML-Lens Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait for backend
timeout /t 3 /nobreak >nul

:: Start frontend
echo.
echo [4/4] Starting frontend...
start "AutoML-Lens Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================
echo   AutoML-Lens is starting!
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Frontend: http://localhost:5173
echo ============================================
echo.
pause
