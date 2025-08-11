@echo off
echo.
echo ========================================
echo  Sales Pipeline Data Explorer
echo ========================================
echo.

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

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
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please delete the 'venv' folder and run update_dependencies.bat again
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Check if Streamlit is installed in virtual environment
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit is not installed in the virtual environment
    echo Please run update_dependencies.bat to install required packages
    echo.
    deactivate
    pause
    exit /b 1
)

echo Starting Sales Pipeline Data Explorer...
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C in this window to stop the application.
echo.

REM Launch the Streamlit app in virtual environment
streamlit run main.py

echo.
echo Application has been stopped.
echo Deactivating virtual environment...
deactivate

pause
