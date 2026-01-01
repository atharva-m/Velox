#!/bin/bash

set -e

echo "Container started"

# Check if model weights already exist
if [ ! -f "model.pth" ]; then
    echo "Model not found. Starting training"
    python train.py
    echo "Training complete. Model saved."
else
    echo "Found existing model. Skipping training."
fi

echo "Starting Consumer Application"

exec python consumer.py