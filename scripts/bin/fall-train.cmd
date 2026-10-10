@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m fall_detection.train %*
exit /b %errorlevel%
