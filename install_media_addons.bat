@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements_media_addons.txt
echo.
echo Media add-ons installed.
pause
