@echo off
"%~dp0..\..\.venv\Scripts\python.exe" -m bounding_box.geometry %*
exit /b %errorlevel%
