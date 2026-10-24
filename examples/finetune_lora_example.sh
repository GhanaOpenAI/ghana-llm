#!/usr/bin/env bash
# Fine-tune Ghana-LLM-Base using LoRA on custom data
set -e

echo "=== Fine-tuning Ghana-LLM-Base with LoRA ==="

cd "$(dirname "$0")/../trainer"

python train_lora.py \
    --from_weight none \
    --data_path ../examples/custom_dataset_sample.jsonl \
    --lora_name lora_custom_twi \
    --epochs 3 \
    --batch_size 16 \
    --accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_seq_len 768 \
    --dtype bfloat16 \
    --save_dir ../out

echo "LoRA fine-tuning complete! Weights saved to ../out/lora_custom_twi_768.pth"
