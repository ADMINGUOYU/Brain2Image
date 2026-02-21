#!/bin/bash

# Experiment Configuration
SEED=648
CUDA=0
EPOCHS=100
BATCH_SIZE=64
LR=1e-4
MULTI_LR="true"
WEIGHT_DECAY=5e-2
OPTIMIZER="AdamW"
CLIP_VALUE=1.0

# Backbone — change this to switch between models
EEG_ENCODER_TYPE="CBraMod"  # Options: "CBraMod", "ATMS"

# Path Configuration (auto-selected by backbone)
if [ "$EEG_ENCODER_TYPE" = "CBraMod" ]; then
    DATASETS_DIR="datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01"
    FOUNDATION_DIR="datasets/processed/cbramod/pretrained_weights.pth"
elif [ "$EEG_ENCODER_TYPE" = "ATMS" ]; then
    DATASETS_DIR="datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01_250Hz"
    FOUNDATION_DIR="datasets/processed/atms/sub-01.pth"
else
    echo "ERROR: Unknown EEG_ENCODER_TYPE=$EEG_ENCODER_TYPE"
    exit 1
fi

# Model Configuration
FROZEN="false"  # Set to "true" to freeze EEG encoder (linear probe)
USE_PRETRAINED_WEIGHTS="true"
MODEL_DIR=""  # Only set if loading full checkpoint

# Classification Head Parameters
NUM_CLASSES=5
HIDDEN_DIM=512
DROPOUT=0.3
MLP_LAYERS=2

# Please configure one of the following
# ---------------------------------------------------- #
# ATMS-specific parameters (if EEG_ENCODER_TYPE=ATMS)
ATMS_EMB_SIZE=40
ATMS_DROP_PROJ=0.5
ATMS_D_MODEL=250
ATMS_N_HEADS=4
ATMS_D_FF=256
ATMS_DROPOUT=0.25
ATMS_FACTOR=1
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
CMD="python -m train.train_EEG_classify \
    --seed $SEED \
    --cuda $CUDA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --multi_lr $MULTI_LR \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --clip_value $CLIP_VALUE \
    --backbone $EEG_ENCODER_TYPE \
    --frozen $FROZEN \
    --use_pretrained_weights $USE_PRETRAINED_WEIGHTS \
    --datasets_dir $DATASETS_DIR \
    --num_classes $NUM_CLASSES \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --mlp_layers $MLP_LAYERS \
    --atms_emb_size $ATMS_EMB_SIZE \
    --atms_drop_proj $ATMS_DROP_PROJ \
    --atms_d_model $ATMS_D_MODEL \
    --atms_n_heads $ATMS_N_HEADS \
    --atms_d_ff $ATMS_D_FF \
    --atms_dropout $ATMS_DROPOUT \
    --atms_factor $ATMS_FACTOR \
    --script_path $0"

# Add conditional arguments
if [ -n "$FOUNDATION_DIR" ]; then
    CMD="$CMD --foundation_dir $FOUNDATION_DIR"
fi

if [ -n "$MODEL_DIR" ]; then
    CMD="$CMD --model_dir $MODEL_DIR"
fi

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
