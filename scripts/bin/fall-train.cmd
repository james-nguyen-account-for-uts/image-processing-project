@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m yolov8.train %*
exit /b %errorlevel%
