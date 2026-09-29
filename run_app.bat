@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.14 -m venv .venv
    if errorlevel 1 goto setup_error
)
call .venv\Scripts\activate.bat
if errorlevel 1 goto setup_error
python -m pip install -r requirements.txt
if errorlevel 1 goto setup_error
python -m streamlit run app.py
pause
exit /b

:setup_error
echo Project setup failed. Verify Python 3.14 is installed and retry.
pause
exit /b 1
