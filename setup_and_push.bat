@echo off
setlocal
echo ==========================================
echo   Karachi AQI Predictor - Fresh Setup
echo ==========================================

:: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. 
    echo Please install it from: https://git-scm.com/downloads
    echo This script needs Git to handle hidden folders like .github automatically.
    pause
    exit /b
)

:: Clean up any existing local git info to start totally fresh
if exist .git (
    echo [0/5] Cleaning up old local git data...
    rmdir /s /q .git
)

echo [1/5] Initializing Fresh Git Repository...
git init

echo [2/5] Adding all files (including hidden folders)...
git add .

echo [3/5] Creating Initial Commit...
git commit -m "Fresh deployment release v1.0"

echo [4/5] Setting main branch...
git branch -M main

echo [5/5] Connecting to GitHub (karankumar56/aqi-predictor-karachi)...
git remote add origin https://github.com/karankumar56/aqi-predictor-karachi.git

echo [6/6] Pushing to GitHub...
git push -u origin main --force

echo.
echo ==========================================
echo   DONE! Check your GitHub page now.
echo ==========================================
pause
