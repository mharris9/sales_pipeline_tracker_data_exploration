@echo off
echo.
echo ========================================
echo  Clean Virtual Environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo No virtual environment found to clean.
    echo.
    pause
    exit /b 0
)

echo WARNING: This will delete the entire virtual environment.
echo You will need to run update_dependencies.bat again to recreate it.
echo.
set /p confirm="Are you sure you want to continue? (y/N): "

if /i "%confirm%" neq "y" (
    echo Operation cancelled.
    echo.
    pause
    exit /b 0
)

echo.
echo Removing virtual environment...

REM Try to remove the directory
rmdir /s /q "venv" 2>nul

if exist "venv\" (
    echo ERROR: Could not remove virtual environment directory
    echo Please close any applications using files in the venv folder
    echo and try again, or manually delete the 'venv' folder
    echo.
) else (
    echo Virtual environment successfully removed.
    echo.
    echo To recreate the environment, run:
    echo   update_dependencies.bat
    echo.
)

pause
