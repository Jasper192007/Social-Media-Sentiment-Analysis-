#!/usr/bin/env bash
# Mac/Linux: creates a virtual environment, installs requirements and starts the app.
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then python3 -m venv venv; fi
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
