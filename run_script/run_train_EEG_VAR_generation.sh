#!/bin/bash

# EEG-VAR Generation Training Script
# Stage 2: VAR-based autoregressive image reconstruction

# Stage 1 checkpoint (from EEG-CLIP alignment training)
# Update this path to your trained stage 1 checkpoint
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

# Training hyperparameters
EPOCHS=150
BATCH_SIZE=64
LR=1e-4
WARMUP_EPOCHS=10
WEIGHT_DECAY=0.05
LABEL_SMOOTHING=0.1
GRAD_CLIP=1.0
FREEZE_EEG_ENCODER="true"       # Set to "false" to unfreeze and train EEG encoder
EEG_ENCODER_LR_SCALE=0.1        # LR multiplier for EEG encoder when unfrozen (0.1 = 10x lower)
UNFROZEN_EEG_MODE="eval"        # "eval" (partial fine-tuning, BN/Dropout frozen) or "train" (full fine-tuning)
                                # Only applies when FREEZE_EEG_ENCODER="false". Default: "eval" (recommended)

# Data loading
NUM_WORKERS=8

# Logging
LOG_INTERVAL=10
VAL_INTERVAL=1
SAVE_INTERVAL=10

# Output directory
OUTPUT_DIR="runs"

# Execute training
python -m train.train_EEG_VAR_generation \
    --stage1_ckpt $STAGE1_CKPT \
    --lmdb_dir $LMDB_DIR \
    --h5_path $H5_PATH \
    --vae_ckpt $VAE_CKPT \
    --var_ckpt $VAR_CKPT \
    --eeg_encoder_type $EEG_ENCODER_TYPE \
    --var_depth $VAR_DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --label_smoothing $LABEL_SMOOTHING \
    --grad_clip $GRAD_CLIP \
    --freeze_eeg_encoder $FREEZE_EEG_ENCODER \
    --eeg_encoder_lr_scale $EEG_ENCODER_LR_SCALE \
    --unfrozen_eeg_mode $UNFROZEN_EEG_MODE \
    --num_workers $NUM_WORKERS \
    --log_interval $LOG_INTERVAL \
    --val_interval $VAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --output_dir $OUTPUT_DIR \
    --device cuda
