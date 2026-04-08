@echo off
REM Start Prometheus and Grafana monitoring stack
REM Requires Docker Desktop to be installed and running

echo ========================================
echo  Sentinel Monitoring Stack
echo ========================================
echo.

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found. Please install Docker Desktop first.
    echo Download from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo Starting monitoring services...
cd /d "%~dp0..\tools"
docker compose -f docker-compose-monitoring.yml up -d

echo.
echo ========================================
echo  Services Started!
echo ========================================
echo  Prometheus:  http://localhost:9090
echo  Grafana:     http://localhost:3000
echo  Grafana login: admin / admin
echo ========================================
echo.
echo To stop: docker compose -f tools\docker-compose-monitoring.yml down
echo.
pause
