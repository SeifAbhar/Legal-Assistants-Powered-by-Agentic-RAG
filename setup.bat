@echo off
echo Legal AI Assistant — Setup
echo ============================

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10+ is required. Please install from https://python.org
    exit /b 1
)

if not exist .venv (
    python -m venv .venv
    echo Virtual environment created.
)

call .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
    copy .env.example .env
    echo .env file created from .env.example — please edit it and add your OPENAI_API_KEY.
) else (
    echo .env file already exists.
)

echo.
echo Setup complete!
echo Next steps:
echo 1. Edit .env and add your OpenAI API key
echo 2. Run: python run.py build
echo 3. Run: python run.py ui