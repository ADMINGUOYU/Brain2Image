#!/bin/bash

# ============================================================================ #
# EEG-CLIP Alignment Training Script
# ============================================================================ #
# This script trains an EEG encoder to align EEG signals with CLIP-H embeddings
# (1024-dim) extracted from images. No fMRI data is used in this pipeline.
#
# Usage:
#   bash run_script/run_train_EEG_CLIP_align.sh
#
# Key differences from EEG-fMRI alignment:
#   - Target: CLIP-H embeddings (1024-dim) instead of fMRI (4096-dim)
#   - Dataset: EEG-only, no paired fMRI data required
#   - Loss: MSE + InfoNCE (no prototypical distillation)
# ============================================================================ #

# ============================================================================ #
# Experiment Configuration
# ============================================================================ #
SEED=648                    # Random seed for reproducibility
CUDA=0                      # GPU device ID
EPOCHS=150                  # Number of training epochs
BATCH_SIZE=128               # Batch size for training
LR=5e-4                     # Base learning rate
MULTI_LR="true"             # Use differential learning rates (backbone: BACKBONE_LR_SCALE x, head: 1.0x)
BACKBONE_LR_SCALE=1         # LR multiplier for eeg_encoder.backbone when MULTI_LR=true
WARMUP_EPOCHS=15            # Linear warmup epochs before cosine annealing (0 = no warmup)
WEIGHT_DECAY=5e-2           # Weight decay for AdamW optimizer
OPTIMIZER="AdamW"           # Optimizer type
CLIP_VALUE=1.0              # Gradient clipping value
EXPERIMENT_FOLDER=""        # Subfolder under ./runs (empty = no subfolder)
EXPERIMENT_NAME=""          # Custom experiment name (empty = default "EEG_CLIP_align")

# ============================================================================ #
# Subject Configuration
# ============================================================================ #
# Change this to train on different THINGS-EEG subjects
THINGS_SUBJECT="sub-08"     # Options: "sub-01", "sub-02", ..., "sub-10"

# ============================================================================ #
# Backbone Configuration
# ============================================================================ #
# EEG encoder backbone — determines input frequency and model architecture
EEG_ENCODER_TYPE="CBraMod"  # Options: "CBraMod" (200Hz), "ATMS" (250Hz)

# ============================================================================ #
# Path Configuration (auto-selected by backbone and subject)
# ============================================================================ #
# Paths are automatically constructed based on:
#   - EEG_ENCODER_TYPE: determines frequency (200Hz for CBraMod, 250Hz for ATMS)
#   - THINGS_SUBJECT: determines which subject's data to use
#   - filter_things_test_split: "_no_things_test" suffix if test split excluded
#
# Dataset structure:
#   datasets/processed/eeg_clip_align_datasets/
#       things_sub-01_clip_no_things_test/          (CBraMod, 200Hz)
#       things_sub-01_clip_no_things_test_250Hz/    (ATMS, 250Hz)
#       things_sub-08_clip_no_things_test/          (CBraMod, 200Hz)
#       things_sub-08_clip_no_things_test_250Hz/    (ATMS, 250Hz)
#
# Foundation model (pretrained weights):
#   datasets/processed/cbramod/pretrained_weights.pth  (CBraMod backbone)
#   datasets/processed/atms/sub-01.pth                 (ATMS for sub-01)
#   datasets/processed/atms/sub-08.pth                 (ATMS for sub-08)

if [ "$EEG_ENCODER_TYPE" = "CBraMod" ]; then
    # CBraMod: 200Hz EEG, 12-layer criss-cross transformer
    DATASETS_DIR="datasets/processed/eeg_clip_align_datasets/things_${THINGS_SUBJECT}_clip_no_things_test"
    FOUNDATION_DIR="datasets/processed/cbramod/pretrained_weights.pth"
elif [ "$EEG_ENCODER_TYPE" = "ATMS" ]; then
    # ATMS: 250Hz EEG, iTransformer + ShallowConvNet
    DATASETS_DIR="datasets/processed/eeg_clip_align_datasets/things_${THINGS_SUBJECT}_clip_no_things_test_250Hz"
    FOUNDATION_DIR="datasets/processed/atms/${THINGS_SUBJECT}.pth"
else
    echo "ERROR: Unknown EEG_ENCODER_TYPE=$EEG_ENCODER_TYPE"
    echo "Valid options: CBraMod, ATMS"
    exit 1
fi

# ============================================================================ #
# Model Configuration
# ============================================================================ #
FROZEN="false"              # Set to "true" to freeze EEG encoder (only train projection head)
USE_PRETRAINED_WEIGHTS="true"  # Load pretrained backbone weights from FOUNDATION_DIR
MODEL_DIR=""                # Path to full checkpoint (leave empty for fresh training)
                            # If set, loads entire model state (overrides FOUNDATION_DIR)

# ============================================================================ #
# Loss Configuration
# ============================================================================ #
# EEG-CLIP alignment uses two loss components:
#   1. MSE Loss: L2 distance between EEG projection and CLIP embedding
#   2. InfoNCE Loss: Contrastive loss for batch-level alignment
#
# Note: No prototypical distillation (no cluster labels in EEG-CLIP dataset)

MSE_SCALE=1.0                    # Weight for MSE loss
INFONCE_SCALE=0.3                # Weight for InfoNCE contrastive loss
TEMPERATURE=0.07                 # Initial temperature for InfoNCE (default: 0.07)
LEARNABLE_TEMPERATURE="true"     # Make temperature learnable during training (default: true)
NORMALIZE_CLIP="true"            # L2-normalize CLIP embeddings before loss computation

# ---------------------------------------------------- #
# ATMS-specific parameters (if EEG_ENCODER_TYPE=ATMS)
ATMS_EMB_SIZE=40
OUT_MLP_DIM=0       # 0 = no extra projection (ATMS outputs 1024 natively)
ATMS_DROP_PROJ=0.5
ATMS_D_MODEL=250
ATMS_N_HEADS=4
ATMS_D_FF=256
ATMS_DROPOUT=0.25
ATMS_FACTOR=1
# ---------------------------------------------------- #
# CBraMod-specific parameters (if EEG_ENCODER_TYPE=CBraMod)
POOLING_TYPE="multitoken_vit"  # Options: "attention", "multitoken_vit", "flatten"
EMBEDDING_DIM=1024
MLP_LAYERS=2
ATTENTION_HEADS=8
NUM_TOKENS=4
NUM_TRANSFORMER_LAYERS=4
NUM_ATTENTION_HEADS=4
# ---------------------------------------------------- #

# Dependency Checks
if [ -n "$MODEL_DIR" ] && [ -n "$FOUNDATION_DIR" ]; then
    echo "ERROR: Cannot specify both --model_dir and --foundation_dir"
    exit 1
fi

if [ "$USE_PRETRAINED_WEIGHTS" = "true" ] && [ -z "$FOUNDATION_DIR" ]; then
    echo "ERROR: Must provide --foundation_dir when using pretrained weights"
    exit 1
fi

# Build Command
CMD="python -m train.train_EEG_CLIP_align \
    --seed $SEED \
    --cuda $CUDA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --multi_lr $MULTI_LR \
    --backbone_lr_scale $BACKBONE_LR_SCALE \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --clip_value $CLIP_VALUE \
    --backbone $EEG_ENCODER_TYPE \
    --frozen $FROZEN \
    --use_pretrained_weights $USE_PRETRAINED_WEIGHTS \
    --datasets_dir $DATASETS_DIR \
    --embedding_dim $EMBEDDING_DIM \
    --mlp_layers $MLP_LAYERS \
    --pooling_type $POOLING_TYPE \
    --attention_heads $ATTENTION_HEADS \
    --num_tokens $NUM_TOKENS \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --atms_emb_size $ATMS_EMB_SIZE \
    --out_mlp_dim $OUT_MLP_DIM \
    --atms_drop_proj $ATMS_DROP_PROJ \
    --atms_d_model $ATMS_D_MODEL \
    --atms_n_heads $ATMS_N_HEADS \
    --atms_d_ff $ATMS_D_FF \
    --atms_dropout $ATMS_DROPOUT \
    --atms_factor $ATMS_FACTOR \
    --mse_scale $MSE_SCALE \
    --infonce_scale $INFONCE_SCALE \
    --temperature $TEMPERATURE \
    --learnable_temperature $LEARNABLE_TEMPERATURE \
    --normalize_clip $NORMALIZE_CLIP \
    --script_path $0"

# Add conditional arguments
if [ -n "$FOUNDATION_DIR" ]; then
    CMD="$CMD --foundation_dir $FOUNDATION_DIR"
fi

if [ -n "$MODEL_DIR" ]; then
    CMD="$CMD --model_dir $MODEL_DIR"
fi

if [ -n "$EXPERIMENT_FOLDER" ]; then
    CMD="$CMD --experiment_folder $EXPERIMENT_FOLDER"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment_name $EXPERIMENT_NAME"
fi

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
