#!/bin/bash
set -e

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install core dependencies first
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-dotenv==1.0.0
pip install pydantic==2.5.0
pip install aiofiles==23.2.0
pip install requests==2.31.0
pip install Pillow==10.1.0
pip install SpeechRecognition==3.10.0
pip install google-generativeai==0.8.3

# Install PyTorch CPU version
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install transformers
pip install transformers==4.35.2

# Install spaCy
pip install spacy==3.7.2

# Download spaCy model
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo "Build completed successfully!"
