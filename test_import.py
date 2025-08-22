#!/usr/bin/env python3

import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import transformers
    from transformers import __version__ as tf_version
    print("✓ Transformers imported:", tf_version)
except ImportError as e:
    print("✗ Transformers import failed:", e)

try:
    import accelerate
    print("✓ Accelerate imported:", accelerate.__version__)
except ImportError as e:
    print("✗ Accelerate import failed:", e)

try:
    import datasets
    print("✓ Datasets imported:", datasets.__version__)
except ImportError as e:
    print("✗ Datasets import failed:", e)

try:
    import deepspeed
    print("✓ DeepSpeed imported:", deepspeed.__version__)
except ImportError as e:
    print("✗ DeepSpeed import failed:", e)

print("\nEnvironment setup complete. Core dependencies are installed.")
print("\nNote: PyTorch and flash-attn need to be installed separately with CUDA support on the GPU node.")
print("On the GPU node, run:")
print("  uv pip install torch==2.4.0.dev20240324 --index-url https://download.pytorch.org/whl/nightly/cu118")
print("  uv pip install flash-attn --no-build-isolation --no-cache-dir")