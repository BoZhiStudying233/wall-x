
#!/usr/bin/env bash
set -euo pipefail

# run_vqa_train_sft.sh

# Example runner for VQA SFT training (prototype)
# Adjust the paths below to your environment before running.

# Path to the pretrained model directory (must be compatible with Qwen2_5_VLMoEForAction)
MODEL_PATH="/inspire/hdd/global_user/konghanlin-253108540238/VLA_pt/wall-oss-fast-withMOE"

# JSONL dataset file (each line contains keys: image, question, answer, question_type)
TRAIN_FILE="/inspire/hdd/global_user/konghanlin-253108540238/VSI_sft/data/VSI-590K/vqa.jsonl"
# Optional validation file (set to same for demo or point to a held-out split)
VAL_FILE="/inspire/hdd/global_user/konghanlin-253108540238/VSI_sft/data/VSI-590K/vqa.jsonl"

# Root directory for resolving image relative paths in JSONL
IMAGE_ROOT="/inspire/hdd/global_user/konghanlin-253108540238/VSI_sft/data/VSI-590K/"

# Output directory for checkpoints
OUTPUT_DIR="/inspire/hdd/global_user/konghanlin-253108540238/new_wallx/wallx_pt/vqa_sft_ckpt"

# Training hyperparameters
BATCH_SIZE=2
EPOCHS=1

STEPS=50000

LR=5e-5
GRAD_ACCUM=1
NUM_WORKERS=16
PRECISION="bf16"   # bf16 | fp16 | fp32
LOG_EVERY=100
VALIDATE_EVERY=500
MAX_IMAGE_SIDE=256
MAX_VAL_SAMPLES=200
SAVE_FORMAT="safetensors"  # pt | safetensors
WARMUP_RATIO=0.05
SAVE_EVERY_STEPS=5000
TRAIN_LOG_PATH="/inspire/hdd/global_user/konghanlin-253108540238/new_wallx/wallx_pt/vqa_sft_ckpt/training_log.txt"

# Use 2 GPUs on a single node by default
export CUDA_VISIBLE_DEVICES=0,1,2,3

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=95

# torchrun will spawn one process per GPU
torchrun --nproc_per_node=4 vqa_train_sft.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --training_steps ${STEPS} \
  --lr ${LR} \
  --grad_accum_steps ${GRAD_ACCUM} \
  --num_workers ${NUM_WORKERS} \
  --precision ${PRECISION} \
  --log_every ${LOG_EVERY} \
  --val_file "${VAL_FILE}" \
  --validate_every ${VALIDATE_EVERY} \
  --max_val_samples ${MAX_VAL_SAMPLES} \
  --save_format ${SAVE_FORMAT} \
  --warmup_ratio ${WARMUP_RATIO} \
  --save_every_steps ${SAVE_EVERY_STEPS} \
  --train_log_path "${TRAIN_LOG_PATH}" \
  --max_image_side "${MAX_IMAGE_SIDE}" \
