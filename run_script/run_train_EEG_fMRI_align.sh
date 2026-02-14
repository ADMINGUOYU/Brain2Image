#!/bin/bash

# Experiment Configuration
SEED=3407
CUDA=0
EPOCHS=100
BATCH_SIZE=64
LR=1e-4
MULTI_LR="true"  # Set to "false" to disable multi-learning rate
WEIGHT_DECAY=5e-2
OPTIMIZER="AdamW"
CLIP_VALUE=1.0
DROPOUT=0.25

# Path Configuration
DATASETS_DIR="/path/to/datasets"

# Model Configuration
FROZEN="false"  # Set to "true" to freeze EEG encoder
USE_PRETRAINED_WEIGHTS="true"
FOUNDATION_DIR="/path/to/foundation/model"  # Required if USE_PRETRAINED_WEIGHTS=true
MODEL_DIR=""  # Only set if loading full checkpoint

# Architecture Parameters
MSE_SCALE=1.0
INFONCE_SCALE=1.0
TEMPERATURE=0.07
NORMALIZE_FMRI="true"
POOLING_TYPE="multitoken_vit"
# --- attention and flatten pooling
EMBEDDING_DIM=4096
MLP_LAYERS=2
# --- attention pooling
ATTENTION_HEADS=16
# --- multitoken_vit pooling
NUM_TOKENS=4
NUM_TRANSFORMER_LAYERS=4
NUM_ATTENTION_HEADS=4
# Alignment attention parameters
ALIGNMENT_ATTENTION_HEADS=4
ALIGNMENT_ATTENTION_DROPOUT=0.25

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
CMD="python train_EEG_fMRI_align.py \
    --seed $SEED \
    --cuda $CUDA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --multi_lr $MULTI_LR \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --clip_value $CLIP_VALUE \
    --frozen $FROZEN \
    --use_pretrained_weights $USE_PRETRAINED_WEIGHTS \
    --datasets_dir $DATASETS_DIR \
    --embedding_dim $EMBEDDING_DIM \
    --mlp_layers $MLP_LAYERS \
    --mse_scale $MSE_SCALE \
    --infonce_scale $INFONCE_SCALE \
    --temperature $TEMPERATURE \
    --normalize_fmri $NORMALIZE_FMRI \
    --pooling_type $POOLING_TYPE \
    --attention_heads $ATTENTION_HEADS \
    --num_tokens $NUM_TOKENS \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --alignment_attention_heads $ALIGNMENT_ATTENTION_HEADS \
    --alignment_attention_dropout $ALIGNMENT_ATTENTION_DROPOUT"

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