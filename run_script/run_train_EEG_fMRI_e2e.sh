#!/bin/bash

# Experiment Configuration
SEED=648
CUDA=0
EPOCHS=150
BATCH_SIZE=64
LR=3e-4
WEIGHT_DECAY=5e-2
CLIP_VALUE=1.0
USE_AMP="true"
NUM_WORKERS=16

# Backbone — change this to switch between models
EEG_ENCODER_TYPE="ATMS"  # Options: "CBraMod", "ATMS"

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

IMAGES_DF_DIR="datasets/processed"

MindEYE2_CKPT_PATH="datasets/processed/mindeye2/sub-01_last_full.pth"

# Model Configuration
FREEZE_ENCODER="false"
FREEZE_MINDEYE2="false"
USE_PRETRAINED_WEIGHTS="true"
MODEL_DIR=""           # Full E2E checkpoint (only set if resuming)
ALIGN_MODEL_DIR=""     # Pretrained alignment checkpoint (warm-start)

# Alignment Loss Parameters
MSE_SCALE=1.0
INFONCE_SCALE=0.2
PROTO_DISTILL_SCALE=0.0
TEMPERATURE=0.1
NORMALIZE_FMRI="true"

# E2E Generation Loss Parameters
ALIGN_SCALE=1.0
PRIOR_SCALE=30.0
CLIP_LOSS_SCALE=1.0
BLUR_SCALE=0.5
MIXUP_PCT=0.33
BLURRY_RECON="true"
EMB_SOURCE="things"  # Options: "nsd", "things" — which image source for generation targets

# Please configure one of the following
# ---------------------------------------------------- #
# ATMS-specific parameters (if EEG_ENCODER_TYPE=ATMS)
ATMS_EMB_SIZE=40
OUT_MLP_DIM=4096
ATMS_DROP_PROJ=0.5
ATMS_D_MODEL=250
ATMS_N_HEADS=4
ATMS_D_FF=256
ATMS_DROPOUT=0.25
ATMS_FACTOR=1
# ---------------------------------------------------- #
# CBraMod-specific parameters (if EEG_ENCODER_TYPE=CBraMod)
POOLING_TYPE="multitoken_vit"  # Options: "attention", "multitoken_vit", "flatten"
# ---  attention and flatten pooling
EMBEDDING_DIM=4096
MLP_LAYERS=2
# --- attention pooling
ATTENTION_HEADS=16
# --- multitoken_vit pooling
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
CMD="python -m train.train_EEG_fMRI_e2e \
    --seed $SEED \
    --cuda $CUDA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --clip_value $CLIP_VALUE \
    --use_amp $USE_AMP \
    --backbone $EEG_ENCODER_TYPE \
    --freeze_encoder $FREEZE_ENCODER \
    --freeze_mindeye2 $FREEZE_MINDEYE2 \
    --use_pretrained_weights $USE_PRETRAINED_WEIGHTS \
    --datasets_dir $DATASETS_DIR \
    --images_df_dir $IMAGES_DF_DIR \
    --num_workers $NUM_WORKERS \
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
    --proto_distill_scale $PROTO_DISTILL_SCALE \
    --temperature $TEMPERATURE \
    --normalize_fmri $NORMALIZE_FMRI \
    --align_scale $ALIGN_SCALE \
    --prior_scale $PRIOR_SCALE \
    --clip_loss_scale $CLIP_LOSS_SCALE \
    --blur_scale $BLUR_SCALE \
    --mixup_pct $MIXUP_PCT \
    --blurry_recon $BLURRY_RECON \
    --emb_source $EMB_SOURCE \
    --script_path $0"

# Add conditional arguments
if [ -n "$FOUNDATION_DIR" ]; then
    CMD="$CMD --foundation_dir $FOUNDATION_DIR"
fi

if [ -n "$MindEYE2_CKPT_PATH" ]; then
    CMD="$CMD --mindeye2_ckpt_path $MindEYE2_CKPT_PATH"
fi

if [ -n "$MODEL_DIR" ]; then
    CMD="$CMD --model_dir $MODEL_DIR"
fi

if [ -n "$ALIGN_MODEL_DIR" ]; then
    CMD="$CMD --align_model_dir $ALIGN_MODEL_DIR"
fi

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
