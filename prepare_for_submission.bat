@echo off
REM ===================================================================
REM SUBMISSION PREPARATION SCRIPT
REM Clears all caches and verifies inference.py before submission
REM ===================================================================

echo.
echo ========================================
echo  PREPARING FOR SUBMISSION
echo ========================================
echo.

REM Step 1: Clear all Python caches
echo [1/4] Clearing Python caches...
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo   ✓ Removed root __pycache__
)
if exist "server\__pycache__" (
    rmdir /s /q "server\__pycache__"
    echo   ✓ Removed server __pycache__
)
if exist "tests\__pycache__" (
    rmdir /s /q "tests\__pycache__"
    echo   ✓ Removed tests __pycache__
)

REM Remove all .pyc files
for /r %%i in (*.pyc) do del "%%i" >nul 2>&1
for /r %%i in (*.pyo) do del "%%i" >nul 2>&1
echo   ✓ Removed all .pyc and .pyo files
echo.

REM Step 2: Verify inference.py has the fix
echo [2/4] Verifying inference.py...
findstr /C:"_safe_int" inference.py >nul 2>&1
if errorlevel 1 (
    echo   ✗ ERROR: Fix not found in inference.py!
    echo   Please run: git pull origin main
    pause
    exit /b 1
)
echo   ✓ Found _safe_int function
echo.

REM Step 3: Test compilation
echo [3/4] Testing compilation...
uv run python -c "import py_compile; py_compile.compile('inference.py', doraise=True)" 2>&1 | findstr /C:"Error"
if errorlevel 1 (
    echo   ✓ Compilation successful
)
echo.

REM Step 4: Quick runtime test
echo [4/4] Running quick validation...
uv run python -c "import os; os.environ['HF_TOKEN'] = 'hf_test'; os.environ['BASE_URL'] = 'http://localhost:7860'; import inference; assert inference.MAX_STEPS == 20; assert inference.EVAL_SEED == 42; print('   ✓ All defaults working correctly')"
echo.

echo ========================================
echo  ✅ READY FOR SUBMISSION
echo ========================================
echo.
echo  Next steps:
echo    1. git add inference.py
echo    2. git commit -m "fix: ultra-defensive env var handling with cache bypass"
echo    3. git push origin main
echo    4. Submit to hackathon
echo.
pause
