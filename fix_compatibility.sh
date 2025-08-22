#!/bin/bash

echo "Fixing compatibility issues for EasyContext..."

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the EasyContext directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Please run setup.sh first"
fi

echo "Uninstalling problematic packages..."
pip uninstall -y ring-flash-attn transformers

echo "Installing compatible transformers version..."
pip install "transformers>=4.44.0"

echo "Installing compatible ring-flash-attn..."
pip install "git+https://github.com/zhuzilin/ring-flash-attention@main"

echo "Verifying installation..."
python -c "
try:
    import transformers
    print('✓ Transformers version:', transformers.__version__)
    
    from transformers.modeling_flash_attention_utils import _flash_supports_window_size
    print('✓ _flash_supports_window_size import successful')
    
    import ring_flash_attn
    print('✓ Ring flash attention imported successfully')
    
    print('\\n✅ All compatibility issues fixed!')
except ImportError as e:
    print('❌ Still have import issues:', e)
    print('\\nTry manually installing:')
    print('pip install transformers==4.46.0')
    print('pip install git+https://github.com/zhuzilin/ring-flash-attention@main')
"

echo "Fix completed!"