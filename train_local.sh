#!/bin/bash

MODEL_PATH=${1:-"/path/to/local/model"}
DATASET_PATH=${2:-"/path/to/local/dataset"}
OUTPUT_DIR=${3:-"./output/local_training"}
SEQ_LENGTH=${4:-32768}
PARALLEL_MODE=${5:-"data_parallel"}
MAX_STEPS=${6:-100}
GRADIENT_ACCUMULATE=${7:-4}
BATCH_SIZE=${8:-1}
LEARNING_RATE=${9:-2e-5}
ROPE_THETA=${10:-1000000}
NUM_GPUS=${11:-8}

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'

echo "Training Configuration:"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Sequence Length: $SEQ_LENGTH"
echo "Parallel Mode: $PARALLEL_MODE"
echo "Max Steps: $MAX_STEPS"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "RoPE Theta: $ROPE_THETA"
echo "Number of GPUs: $NUM_GPUS"

CONFIG_FILE="accelerate_configs/single_node.yaml"
if [ "$NUM_GPUS" -gt 8 ]; then
    CONFIG_FILE="accelerate_configs/two_node.yaml"
fi

if [ "$SEQ_LENGTH" -gt 256000 ]; then
    if [ "$PARALLEL_MODE" == "data_parallel" ]; then
        echo "Warning: For sequences > 256K, switching to zigzag_ring_attn for memory efficiency"
        PARALLEL_MODE="zigzag_ring_attn"
    fi
fi

accelerate launch \
    --num_processes $NUM_GPUS \
    --config_file $CONFIG_FILE \
    train.py \
    --batch-size $BATCH_SIZE \
    --gradient-accumulate-every $GRADIENT_ACCUMULATE \
    --output-dir $OUTPUT_DIR \
    --max-train-steps $MAX_STEPS \
    --learning-rate $LEARNING_RATE \
    --dataset $DATASET_PATH \
    --model $MODEL_PATH \
    --seq-length $SEQ_LENGTH \
    --rope-theta $ROPE_THETA \
    --parallel_mode $PARALLEL_MODE

echo "Training completed!"