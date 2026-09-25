:: run.bat — one command to go live
@echo off
cd /d "%~dp0"
set HF_HOME=%~dp0models\hf
:loop
uv run --frozen aivtube run %*
if %errorlevel%==3 goto loop
pause
