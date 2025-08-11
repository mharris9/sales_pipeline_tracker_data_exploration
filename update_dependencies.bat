@echo off
echo.
echo ========================================
echo  Update Dependencies (Virtual Environment)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure you have Python 3.8+ with venv module installed
        echo.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
) else (
    echo Virtual environment already exists.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Try deleting the 'venv' folder and running this script again
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Upgrade pip in virtual environment
echo Updating pip to latest version...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

echo.
echo Installing/updating required packages in virtual environment...
echo This may take a few minutes...
echo.

REM Install requirements in virtual environment
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo Please check the error messages above
    echo.
    echo The virtual environment is still active.
    echo You can try manual installation with:
    echo   pip install package_name
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo  SUCCESS: All dependencies installed!
    echo ========================================
    echo.
    echo Virtual environment location: %cd%\venv
    echo Python location: %VIRTUAL_ENV%\Scripts\python.exe
    echo.
    echo You can now run the application with:
    echo   run_app.bat
    echo.
    echo Or manually activate the environment and run:
    echo   venv\Scripts\activate
    echo   streamlit run main.py
    echo.
)

REM Deactivate virtual environment
deactivate

pause
