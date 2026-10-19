@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m yolov8.detect %*
exit /b %errorlevel%
