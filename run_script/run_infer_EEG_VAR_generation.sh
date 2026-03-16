#!/bin/bash

# EEG-VAR Generation Inference Script
# Generate images from EEG using trained VAR model
# NOTE: Evaluation metrics are computed automatically after generation.
#       Use --skip_eval true to disable evaluation (faster, images only).

# Trained model checkpoint
# Update this path to your trained checkpoint
CHECKPOINT="runs/EEG_VAR_generation_CBraMod_d16_20260316_163820/checkpoints/best_model.pth"

# Extract experiment directory from checkpoint path
EXPERIMENT_DIR="${CHECKPOINT%/checkpoints/*}"

# Stage 1 checkpoint (same as used in training)
STAGE1_CKPT="runs/EEG_CLIP_align_1773643388/checkpoints/best_model_epoch_93.pth"

# Subject configuration
THINGS_SUBJECT="sub-08"
EEG_ENCODER_TYPE="CBraMod"  # or "ATMS"

# Dataset paths
if [ "$EEG_ENCODER_TYPE" = "CBraMod" ]; then
    LMDB_DIR="datasets/processed/eeg_clip_align_datasets/things_${THINGS_SUBJECT}_clip_no_things_test"
elif [ "$EEG_ENCODER_TYPE" = "ATMS" ]; then
    LMDB_DIR="datasets/processed/eeg_clip_align_datasets/things_${THINGS_SUBJECT}_clip_no_things_test_250Hz"
fi

# HDF5 file for images
H5_PATH="datasets/processed/things_images_only.h5"

# Pre-trained checkpoints
VAE_CKPT="datasets/pretrain_weights/vae_ch160v4096z32.pth"
VAR_DEPTH=16
VAR_CKPT="datasets/pretrain_weights/var_d${VAR_DEPTH}.pth"

# Inference config
SPLIT="test"
BATCH_SIZE=16
CFG_SCALE=1.5
TOP_K=900
TOP_P=0.95
SEED=42
MAX_SAMPLES=100  # Set to null to process all samples

# Evaluation config
SKIP_EVAL=false  # Set to true to skip evaluation (faster, images only)
EVAL_BATCH_SIZE=40  # Batch size for evaluation model forward passes

# Output directory (saved in same directory as checkpoint)
OUTPUT_DIR="${EXPERIMENT_DIR}/infer"

# Execute inference
python -m infer.infer_EEG_VAR_generation \
    --checkpoint $CHECKPOINT \
    --stage1_ckpt $STAGE1_CKPT \
    --lmdb_dir $LMDB_DIR \
    --h5_path $H5_PATH \
    --vae_ckpt $VAE_CKPT \
    --var_ckpt $VAR_CKPT \
    --eeg_encoder_type $EEG_ENCODER_TYPE \
    --var_depth $VAR_DEPTH \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --cfg_scale $CFG_SCALE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --seed $SEED \
    --max_samples $MAX_SAMPLES \
    --skip_eval $SKIP_EVAL \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --device cuda
