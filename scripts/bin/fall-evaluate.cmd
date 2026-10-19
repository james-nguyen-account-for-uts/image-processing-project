@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m yolov8.evaluate %*
exit /b %errorlevel%
