@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo  ChatGPT Gemini Media Studio - Charon + FFmpeg
echo ================================================

echo.
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found in PATH.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ERROR: FFmpeg was not found in PATH.
  echo Install FFmpeg and reopen this window.
  pause
  exit /b 1
)

if not exist ".env" (
  if exist ".env.media.example" (
    copy ".env.media.example" ".env" >nul
    echo Created .env from .env.media.example
    echo Edit .env and add your GEMINI_API_KEY / PUBLIC_URL before ChatGPT connection.
    echo.
  )
)

python server_mcp.py
pause
