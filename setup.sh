#!/bin/bash

echo "Setting up EasyContext environment for GPU node..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Creating virtual environment..."
uv venv

echo "Activating environment..."
source .venv/bin/activate

echo "Installing base dependencies..."
uv sync

echo "Installing PyTorch nightly with CUDA 11.8..."
uv pip install torch==2.4.0.dev20240324 --index-url https://download.pytorch.org/whl/nightly/cu118

echo "Installing flash-attn (this may take a while)..."
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation --no-cache-dir

echo "Installing ring-flash-attn..."
uv pip install git+https://github.com/zhuzilin/ring-flash-attention

echo "Setup complete! Activate the environment with: source .venv/bin/activate"
echo "To test the setup, run: python test_import.py"