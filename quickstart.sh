#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
echo 'Environment ready. Launching Jupyter Lab...'
jupyter lab
