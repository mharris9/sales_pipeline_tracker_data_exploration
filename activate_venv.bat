@echo off
echo.
echo ========================================
echo  Activate Virtual Environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found
    echo Please run update_dependencies.bat first to set up the environment
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment has Python
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment appears corrupted
    echo Please delete the 'venv' folder and run update_dependencies.bat again
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
echo.
echo Virtual environment location: %cd%\venv
echo.
echo You can now run Python commands in the isolated environment.
echo.
echo Common commands:
echo   streamlit run main.py    - Run the application
echo   python test_app.py       - Run tests
echo   pip list                 - Show installed packages
echo   pip install package_name - Install additional packages
echo   deactivate               - Exit virtual environment
echo.
echo ========================================

REM Activate and keep the command prompt open
call venv\Scripts\activate.bat && cmd /k
