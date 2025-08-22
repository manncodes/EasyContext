# Local Training Setup for EasyContext

This setup allows you to train long-context models using local model paths and datasets instead of HuggingFace tags.

## Installation

### Using uv (Recommended)
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### Manual Installation
```bash
uv venv
source .venv/bin/activate
uv pip install torch==2.4.0.dev20240324 --index-url https://download.pytorch.org/whl/nightly/cu118
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv sync
```

## Usage

### Testing on Small GPU (Laptop)
```bash
chmod +x test_small.sh
./test_small.sh /path/to/model /path/to/dataset
```

### Training on GPU Node
```bash
chmod +x train_local.sh
./train_local.sh /path/to/model /path/to/dataset ./output/my_training
```

### Full Parameter Control
```bash
./train_local.sh \
    /path/to/model \
    /path/to/dataset \
    ./output/dir \
    32768 \              # sequence length
    data_parallel \      # parallel mode
    1000 \               # max steps
    4 \                  # gradient accumulation
    1 \                  # batch size
    2e-5 \               # learning rate
    1000000 \            # rope theta
    8                    # number of GPUs
```

## Parallel Modes

- `data_parallel`: Standard data parallelism (for sequences up to 64K)
- `zigzag_ring_attn`: ZigZag ring attention (for sequences 256K+)
- `dist_flash_attn`: Distributed flash attention
- `ulysses_attn`: Ulysses attention
- `usp_attn`: Unified sequence parallelism

## Dataset Format

Your local dataset should be in HuggingFace format with an `input_ids` column. You can prepare it using:

```python
from datasets import Dataset, DatasetDict
import torch

data = {
    "input_ids": [tokenized_sequence_1, tokenized_sequence_2, ...]
}
dataset = Dataset.from_dict(data)
dataset.save_to_disk("/path/to/dataset")
```

## Memory Recommendations

- **8K-32K sequences**: 1-2 GPUs with 24GB+ VRAM
- **64K sequences**: 2-4 GPUs with 40GB+ VRAM
- **256K sequences**: 8 GPUs with 40GB+ VRAM (use ring attention)
- **512K-1M sequences**: 8-16 GPUs with 80GB+ VRAM (use ring attention)

## Tips

1. Start with small sequence lengths and gradually increase
2. Use gradient accumulation to simulate larger batch sizes
3. For sequences > 256K, always use ring attention modes
4. Monitor GPU memory usage and adjust batch size accordingly
5. Set `PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'` for better memory management