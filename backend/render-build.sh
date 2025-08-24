#!/usr/bin/env bash
set -euo pipefail
echo ">>> Python version:"
python --version
which python
echo ">>> Pip version:"
pip --version
# 1) Install Python deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Download the model (once, at build time)
mkdir -p model
if [ ! -f model/mnist_cnn.keras ]; then
  echo "Downloading MNIST model..."
  curl -L "$MODEL_URL" -o model/mnist_cnn.keras
fi

# Optional: verify checksum if you set one
if [ -n "${MODEL_SHA256:-}" ]; then
  echo "Verifying checksum..."
  EXPECTED="$(echo "$MODEL_SHA256" | tr '[:upper:]' '[:lower:]')"
  ACTUAL="$(sha256sum model/mnist_cnn.keras | awk '{print tolower($1)}')"
  if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "Checksum mismatch!"
    echo "Expected: $EXPECTED"
    echo "Actual:   $ACTUAL"
    exit 1
  fi
fi


echo "Build step complete."
