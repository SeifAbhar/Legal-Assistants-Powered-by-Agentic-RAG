#!/bin/bash
echo "Legal AI Assistant — Setup"
echo "============================"

if ! command -v python3 &> /dev/null; then
    echo "Python 3.10+ is required. Please install it from https://python.org"
    exit 1
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created."
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env file created from .env.example — please edit it and add your OPENAI_API_KEY."
else
    echo ".env file already exists."
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Run: python run.py build"
echo "3. Run: python run.py ui"