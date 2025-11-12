#!/usr/bin/env bash
set -euo pipefail

# Example runner for VQA SFT training (prototype)
# Adjust the paths below to your environment before running.

# Path to the pretrained model directory (must be compatible with Qwen2_5_VLMoEForAction)
MODEL_PATH="/data2/konghanlin/new_wallx/ori_model/wall-oss-fast-withMOE"

# JSONL dataset file (each line contains keys: image, question, answer, question_type)
TRAIN_FILE="/data2/konghanlin/vqa/hypersim10k.jsonl"
# Optional validation file (set to same for demo or point to a held-out split)
VAL_FILE="/data2/konghanlin/vqa/hypersim10k.jsonl"

# Root directory for resolving image relative paths in JSONL
IMAGE_ROOT="/data2/konghanlin/vqa/"

# Output directory for checkpoints
OUTPUT_DIR="/data2/konghanlin/new_wallx/vqa_sft_ckpt"

# Training hyperparameters
BATCH_SIZE=1
EPOCHS=1
LR=1e-5
GRAD_ACCUM=1
NUM_WORKERS=2
PRECISION="bf16"   # bf16 | fp16 | fp32
LOG_EVERY=10
VALIDATE_EVERY=100
MAX_VAL_SAMPLES=64

python vqa_train_sft.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --grad_accum_steps ${GRAD_ACCUM} \
  --num_workers ${NUM_WORKERS} \
  --precision ${PRECISION} \
  --log_every ${LOG_EVERY} \
  --val_file "${VAL_FILE}" \
  --validate_every ${VALIDATE_EVERY} \
  --max_val_samples ${MAX_VAL_SAMPLES}
