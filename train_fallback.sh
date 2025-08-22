#!/bin/bash

MODEL_PATH=${1:-"/path/to/local/model"}
DATASET_PATH=${2:-"/path/to/local/dataset"}
OUTPUT_DIR=${3:-"./output/local_training"}
SEQ_LENGTH=${4:-32768}
MAX_STEPS=${5:-100}
GRADIENT_ACCUMULATE=${6:-4}
BATCH_SIZE=${7:-1}
LEARNING_RATE=${8:-2e-5}
ROPE_THETA=${9:-1000000}
NUM_GPUS=${10:-8}

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'

echo "=== FALLBACK TRAINING MODE (No Ring Attention) ==="
echo "Training Configuration:"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Sequence Length: $SEQ_LENGTH"
echo "Max Steps: $MAX_STEPS"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "RoPE Theta: $ROPE_THETA"
echo "Number of GPUs: $NUM_GPUS"

if [ "$SEQ_LENGTH" -gt 65536 ]; then
    echo "WARNING: Sequence length > 65K without ring attention may cause OOM"
    echo "Consider reducing sequence length or fixing ring attention compatibility"
fi

CONFIG_FILE="accelerate_configs/single_node.yaml"
if [ "$NUM_GPUS" -gt 8 ]; then
    CONFIG_FILE="accelerate_configs/two_node.yaml"
fi

accelerate launch \
    --num_processes $NUM_GPUS \
    --config_file $CONFIG_FILE \
    train_no_ring.py \
    --batch-size $BATCH_SIZE \
    --gradient-accumulate-every $GRADIENT_ACCUMULATE \
    --output-dir $OUTPUT_DIR \
    --max-train-steps $MAX_STEPS \
    --learning-rate $LEARNING_RATE \
    --dataset $DATASET_PATH \
    --model $MODEL_PATH \
    --seq-length $SEQ_LENGTH \
    --rope-theta $ROPE_THETA

echo "Training completed!"