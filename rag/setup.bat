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

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo X Docker Compose is not available. Please install Docker Compose and try again.
    exit /b 1
)
echo [OK] Docker Compose is available

REM Check if .env file exists in project root
cd ..
if not exist .env (
    echo.
    echo [!] .env file not found in project root
    set /p api_key="Enter your Google API Key: "
    echo GOOGLE_API_KEY=!api_key! > .env
    echo [OK] .env file created
) else (
    echo [OK] .env file exists
)
cd rag

REM Start Weaviate using Docker Compose
echo.
echo [*] Starting Weaviate with Docker Compose...
docker-compose up -d
if errorlevel 1 (
    echo X Failed to start Weaviate
    exit /b 1
)
echo [OK] Weaviate started successfully
echo Waiting for Weaviate to be ready...
timeout /t 5 /nobreak >nul

REM Check if dependencies are installed
echo.
echo [*] Checking Python dependencies...
python -c "import weaviate" 2>nul
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
echo Weaviate is running at: http://localhost:8080
echo Check status: http://localhost:8080/v1/meta
echo.
echo To stop Weaviate: docker-compose down
pause

