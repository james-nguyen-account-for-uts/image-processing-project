@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m fall_detection.evaluate %*
exit /b %errorlevel%
