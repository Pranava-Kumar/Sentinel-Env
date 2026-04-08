@echo off
REM ============================================================
REM SENTINEL ENVIRONMENT — One-Click HF Space Deployment
REM ============================================================
REM Prerequisites:
REM   1. huggingface_hub installed: pip install huggingface_hub
REM   2. Logged in: huggingface-cli login
REM   3. Git installed and in PATH
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo  SENTINEL ENVIRONMENT - HF Space Deploy
echo ============================================
echo.

REM Check if logged in
echo [1/6] Checking HuggingFace login...
huggingface-cli whoami >nul 2>&1
if errorlevel 1 (
    echo ERROR: Not logged in to HuggingFace!
    echo Run: huggingface-cli login
    echo Then paste your token from https://huggingface.co/settings/tokens
    pause
    exit /b 1
)
echo OK - Logged in

REM Ask for space name
echo.
echo [2/6] Enter your HF username (or press Enter to auto-detect):
set /p HF_USERNAME=
if "!HF_USERNAME!"=="" (
    for /f "tokens=*" %%i in ('huggingface-cli whoami 2^>nul') do set HF_USERNAME=%%i
)
echo Using username: !HF_USERNAME!

echo.
echo [3/6] Creating HF Space: !HF_USERNAME!/sentinel-env
huggingface-cli repo create sentinel-env --type space --space_sdk docker 2>&1
if errorlevel 1 (
    echo Space may already exist, continuing with update...
)

echo.
echo [4/6] Cloning Space repository...
if exist hf-space-deploy (
    echo Removing old deploy directory...
    rmdir /s /q hf-space-deploy
)
git clone https://huggingface.co/spaces/!HF_USERNAME!/sentinel-env hf-space-deploy
if errorlevel 1 (
    echo ERROR: Failed to clone space. Check that the space was created.
    echo Visit: https://huggingface.co/spaces/!HF_USERNAME!/sentinel-env
    pause
    exit /b 1
)

echo.
echo [5/6] Copying project files to Space...
cd hf-space-deploy

REM Copy the space README (with HF metadata)
copy /y ..\hf-space-readme.md README.md >nul

REM Copy all project files
copy /y ..\Dockerfile . >nul
copy /y ..\.dockerignore . >nul
copy /y ..\models.py . >nul
copy /y ..\client.py . >nul
copy /y ..\openenv.yaml . >nul
copy /y ..\pyproject.toml . >nul
copy /y ..\__init__.py . >nul
copy /y ..\inference.py . >nul
copy /y ..\README.md README-app.md >nul

REM Copy server directory
if exist server rmdir /s /q server
xcopy /E /I /Y /Q ..\server server >nul

REM Create .gitignore for the space
echo __pycache__> .gitignore
echo *.pyc>> .gitignore
echo .env>> .gitignore

echo.
echo [6/6] Committing and pushing to HF...
git add -A
git commit -m "Deploy Sentinel Environment v1.0"
git push

echo.
echo ============================================
echo  DEPLOYMENT COMPLETE!
echo ============================================
echo.
echo  Space URL: https://huggingface.co/spaces/!HF_USERNAME!/sentinel-env
echo.
echo  The Space is now building. This takes 3-5 minutes.
echo  Visit the URL above to check status.
echo.
echo  When it shows "Running", test with:
echo    curl -X POST https://!HF_USERNAME!-sentinel-env.hf.space/reset
echo.

cd ..

pause
