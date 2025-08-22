# GPU Node Setup Guide for EasyContext

## Prerequisites
- GPU node with CUDA 11.8+ installed
- Python 3.10+
- Internet connection for initial setup

## Setup Instructions

### 1. Clone and Navigate to Repository
```bash
cd /path/to/repo/EasyContext
```

### 2. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install uv package manager
- Create virtual environment
- Install all dependencies including PyTorch and flash-attn
- Configure the environment for long-context training

### 3. Activate Environment
```bash
source .venv/bin/activate
```

## Running Training

### Basic Usage
```bash
./train_local.sh /path/to/model /path/to/dataset ./output/training_run
```

### Advanced Usage with Custom Parameters
```bash
./train_local.sh \
    /path/to/llama-7b \           # Model path
    /path/to/tokenized_dataset \   # Dataset path
    ./output/my_training \          # Output directory
    32768 \                         # Sequence length
    data_parallel \                 # Parallel mode
    1000 \                          # Max training steps
    4 \                             # Gradient accumulation steps
    1 \                             # Batch size
    2e-5 \                          # Learning rate
    1000000 \                       # RoPE theta
    8                               # Number of GPUs
```

## Parallel Modes by Sequence Length

| Sequence Length | Recommended Mode | GPU Memory Required |
|----------------|------------------|-------------------|
| 8K-32K | data_parallel | 40GB per GPU |
| 64K | data_parallel | 80GB per GPU |
| 256K | zigzag_ring_attn | 80GB per GPU |
| 512K | zigzag_ring_attn | 80GB per GPU |
| 700K-1M | zigzag_ring_attn | 80GB per GPU |

## Preparing Local Dataset

Your dataset should be in HuggingFace format with tokenized sequences:

```python
from datasets import Dataset
import torch
from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Tokenize your texts
texts = ["your text data here...", ...]
tokenized = [tokenizer(text)["input_ids"] for text in texts]

# Create and save dataset
dataset = Dataset.from_dict({"input_ids": tokenized})
dataset.save_to_disk("/path/to/dataset")
```

## Memory Optimization Tips

1. **Set CUDA memory allocation**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'
   ```

2. **Use gradient accumulation** for larger effective batch sizes:
   ```bash
   --gradient-accumulate-every 8
   ```

3. **Use ring attention** for sequences > 256K:
   ```bash
   --parallel_mode zigzag_ring_attn
   ```

## Monitoring Training

Training logs will be saved to the output directory. If using wandb:
```bash
wandb login  # One-time setup
./train_local.sh ... --wandb your_project_name
```

## Troubleshooting

### Out of Memory
- Reduce sequence length
- Increase gradient accumulation steps
- Switch to ring attention mode
- Reduce batch size to 1

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Ensure flash-attn is properly installed
- Use data_parallel for shorter sequences

### Installation Issues
- Ensure CUDA toolkit matches PyTorch version (11.8)
- Check Python version (3.10+)
- Try manual installation if setup script fails

## Example Training Commands

### Small Test (verify setup)
```bash
./train_local.sh \
    meta-llama/Llama-2-7b-hf \
    emozilla/pg_books-tokenized-bos-eos-chunked-65536 \
    ./output/test \
    8192 \
    data_parallel \
    10 \
    1 \
    1 \
    2e-5 \
    100000 \
    1
```

### Production Training (256K context)
```bash
./train_local.sh \
    /data/models/llama-7b \
    /data/datasets/my_dataset \
    ./output/production_256k \
    256000 \
    zigzag_ring_attn \
    1000 \
    4 \
    1 \
    2e-5 \
    10000000 \
    8
```

## Post-Training

After training completes:
1. Model will be saved in output directory
2. Remove unnecessary files: `rm output/*/model.safetensors`
3. Test the model with eval scripts if needed