#!/usr/bin/env bash
set -euo pipefail

# 1) Install Python deps
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
  ACTUAL=$(sha256sum model/mnist_cnn.keras | awk '{print $1}')
  if [ "$ACTUAL" != "$MODEL_SHA256" ]; then
    echo "Checksum mismatch!"
    echo "Expected: $MODEL_SHA256"
    echo "Actual:   $ACTUAL"
    exit 1
  fi
fi

echo "Build step complete."
