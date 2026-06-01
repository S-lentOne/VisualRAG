#!/usr/bin/env bash

set -e

echo "==> Creating Python environment"

python -m venv bin

source bin/bin/activate

echo "==> Installing Python dependencies"

pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing frontend dependencies"

cd frontend
npm install

echo "==> Setup complete"
