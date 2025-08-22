#!/bin/bash

echo "Test script for small GPU (laptop)"
echo "This will run a minimal test to verify the setup"

MODEL_PATH=${1:-"meta-llama/Llama-2-7b-hf"}
DATASET_PATH=${2:-"emozilla/pg_books-tokenized-bos-eos-chunked-65536"}
OUTPUT_DIR="./output/test_run"

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'

if [ -f "$MODEL_PATH" ] || [ -d "$MODEL_PATH" ]; then
    echo "Using local model at: $MODEL_PATH"
else
    echo "Using HuggingFace model: $MODEL_PATH"
fi

if [ -f "$DATASET_PATH" ] || [ -d "$DATASET_PATH" ]; then
    echo "Using local dataset at: $DATASET_PATH"
else
    echo "Using HuggingFace dataset: $DATASET_PATH"
fi

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train.py \
    --batch-size 1 \
    --gradient-accumulate-every 1 \
    --output-dir $OUTPUT_DIR \
    --max-train-steps 2 \
    --learning-rate 2e-5 \
    --dataset $DATASET_PATH \
    --model $MODEL_PATH \
    --seq-length 2048 \
    --rope-theta 100000 \
    --parallel_mode data_parallel

echo "Test completed! Check $OUTPUT_DIR for results"