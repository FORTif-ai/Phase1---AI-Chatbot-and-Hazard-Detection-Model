@echo off
REM Fortif.ai RAG System - Setup Script (Windows)

echo === Fortif.ai RAG System Setup ===
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo X Docker is not running. Please start Docker and try again.
    exit /b 1
)
echo [OK] Docker is running

REM Check if Qdrant container exists
docker ps -a --format "{{.Names}}" | findstr /x "qdrant" >nul 2>&1
if not errorlevel 1 (
    echo [!] Qdrant container already exists
    set /p restart="Do you want to restart it? (y/n): "
    if /i "%restart%"=="y" (
        docker restart qdrant
        echo [OK] Qdrant container restarted
    )
) else (
    echo [*] Starting Qdrant container...
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
    echo [OK] Qdrant container started
    timeout /t 3 /nobreak >nul
)

REM Check if .env file exists
if not exist .env (
    echo.
    echo [!] .env file not found
    set /p api_key="Enter your Google API Key: "
    echo GOOGLE_API_KEY=!api_key! > .env
    echo [OK] .env file created
) else (
    echo [OK] .env file exists
)

REM Check if dependencies are installed
echo.
echo [*] Checking Python dependencies...
python -c "import qdrant_client" 2>nul
if errorlevel 1 (
    echo [!] Installing dependencies...
    pip install -r ..\requirements.txt
) else (
    echo [OK] Dependencies appear to be installed
)

echo.
echo [SUCCESS] Setup complete!
echo.
echo Next steps:
echo   1. Run: python ingest.py   (to ingest sample data)
echo   2. Run: python query.py    (to test queries)
echo.
echo Qdrant Dashboard: http://localhost:6333/dashboard
pause

