@echo off
setlocal enabledelayedexpansion

echo =================================================================
echo   TrustLens AI - Enterprise ML Platform
echo   Production-Ready Status: COMPLETE
echo =================================================================
echo.

:: Detect workspace root
set ROOT_DIR=%~dp0
cd /d %ROOT_DIR%

echo [1/3] Validating Environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at %ROOT_DIR%.venv
    echo Please run: python -m venv .venv
    pause
    exit /b
)

if not exist "backend\ml\models\fake_review_model.joblib" (
    echo [WARNING] ML models not found in backend\ml\models.
    echo Running inference will use rule-based fallbacks.
)

echo [2/3] Starting Backend (FastAPI on port 8001)...
:: Using start to run in a new window
start "TrustLens Backend" cmd /k "cd /d !ROOT_DIR!backend && ..\.venv\Scripts\python.exe main.py"

echo [3/3] Starting Frontend (Vite on port 5175)...
timeout /t 5 /nobreak >nul
start "TrustLens Frontend" cmd /k "cd /d !ROOT_DIR!frontend && npm run dev"

echo.
echo =================================================================
echo   SUCCESS: TrustLens services are initializing.
echo.
echo   API Health:    http://127.0.0.1:8001/health/full
echo   Prometheus:    http://127.0.0.1:8001/metrics
echo   Web Interface: http://localhost:5175
echo =================================================================
echo.
echo Please wait ~10 seconds for the ML pipeline to load joblib artifacts.
echo You can then paste an Amazon.in URL into the dashboard.
echo.
pause
