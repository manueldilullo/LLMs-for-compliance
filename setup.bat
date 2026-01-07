@echo off
REM Setup script for Windows - LLMs-for-compliance
REM Quick setup using Python setup.py

setlocal

echo ===============================================================
echo   LLMs-for-compliance - Windows Setup
echo ===============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    exit /b 1
)

REM Show menu
:MENU
echo.
echo Choose installation type:
echo   1. Auto-detect and install (recommended)
echo   2. CPU-only installation
echo   3. GPU installation (requires CUDA)
echo   4. Install optional packages (Datapizza, Unsloth)
echo   5. Install development tools
echo   6. Check system
echo   7. Test installation
echo   0. Exit
echo.

set /p choice="Enter choice: "

if "%choice%"=="1" goto AUTO_INSTALL
if "%choice%"=="2" goto CPU_INSTALL
if "%choice%"=="3" goto GPU_INSTALL
if "%choice%"=="4" goto OPTIONAL
if "%choice%"=="5" goto DEV_INSTALL
if "%choice%"=="6" goto CHECK
if "%choice%"=="7" goto TEST
if "%choice%"=="0" goto END
goto MENU

:AUTO_INSTALL
echo.
echo Installing with auto-detection...
python setup.py install
goto MENU

:CPU_INSTALL
echo.
echo Installing CPU-only packages...
python setup.py install-cpu
goto MENU

:GPU_INSTALL
echo.
echo Installing GPU-accelerated packages...
python setup.py install-gpu
goto MENU

:OPTIONAL
echo.
echo Installing optional packages...
python setup.py install-optional
goto MENU

:DEV_INSTALL
echo.
echo Installing development tools...
python setup.py install-dev
goto MENU

:CHECK
echo.
echo Checking system...
python setup.py check
goto MENU

:TEST
echo.
echo Testing installation...
python setup.py test
echo.
echo Testing custom modules...
python test_init_llm.py
python test_datapizza_rag.py
goto MENU

:END
echo.
echo Setup script finished.
endlocal
