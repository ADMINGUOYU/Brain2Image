#!/bin/bash
# Inference script for the ATMS E2E run:
# runs/EEG_fMRI_e2e_1771940569_BRAIN2IMAGE-E2E-ATMS-batchsize-24_NO_blur
#
# Run from the project root:
#   bash run_script/run_infer_EEG_fMRI_e2e.sh

# ── Cache ─────────────────────────────────────────────────────────────────────
export HF_HUB_CACHE="datasets/transformers_cache"
export HUGGINGFACE_HUB_CACHE="datasets/transformers_cache"
export TRANSFORMERS_CACHE="datasets/transformers_cache"

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="runs/EEG_fMRI_e2e_1771940569_BRAIN2IMAGE-E2E-ATMS-batchsize-24_NO_blur"
MODEL_DIR="${EXPERIMENT_DIR}/checkpoints/best_model_epoch_1.pth"
OUTPUT_DIR="${EXPERIMENT_DIR}/inference"

UNCLIP_CKPT="datasets/processed/mindeye2/unclip6_epoch0_step110000.ckpt"
UNCLIP_CONFIG="MindEyeV2/src/generative_models/configs/unclip6.yaml"
AUTOENC_CKPT="datasets/processed/mindeye2/sd_image_var_autoenc.pth"

# ── Inference settings ────────────────────────────────────────────────────────
SPLIT="test"
PRIOR_TIMESTEPS=20
NUM_SAMPLES=1         # num of figure reconstructions to save per sample 
SAVE_BLURRY="false"   # model was trained without blurry recon (NO_blur run)
MAX_BATCHES="10"        # leave empty to run the full split

# ── Hardware ──────────────────────────────────────────────────────────────────
CUDA=0
BATCH_SIZE=8
NUM_WORKERS=16

# ── Backbone — ATMS (must match the training run) ─────────────────────────────
EEG_ENCODER_TYPE="ATMS"
DATASETS_DIR="datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01_250Hz"
IMAGES_DF_DIR="datasets/processed"
NORMALIZE_FMRI="true"
EMB_SOURCE="things"

# ATMS-specific parameters (copied from training run_script.sh)
ATMS_EMB_SIZE=40
OUT_MLP_DIM=4096
ATMS_DROP_PROJ=0.5
ATMS_D_MODEL=250
ATMS_N_HEADS=4
ATMS_D_FF=256
ATMS_DROPOUT=0.25
ATMS_FACTOR=1

# CBraMod parameters (unused for ATMS, kept for reference)
POOLING_TYPE="multitoken_vit"
EMBEDDING_DIM=4096
MLP_LAYERS=2
ATTENTION_HEADS=16
NUM_TOKENS=4
NUM_TRANSFORMER_LAYERS=4
NUM_ATTENTION_HEADS=4

# ── Alignment / loss params (used only for model construction) ────────────────
MSE_SCALE=1.0
INFONCE_SCALE=0.2
PROTO_DISTILL_SCALE=0.0
TEMPERATURE=0.1

# ── Validation ────────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_DIR" ]; then
    echo "ERROR: checkpoint not found at $MODEL_DIR"
    exit 1
fi
if [ ! -f "$UNCLIP_CKPT" ]; then
    echo "ERROR: unCLIP checkpoint not found at $UNCLIP_CKPT"
    echo "  Download with:"
    echo "  wget 'https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/unclip6_epoch0_step110000.ckpt' \\"
    echo "       -O $UNCLIP_CKPT"
    exit 1
fi
if [ ! -f "$UNCLIP_CONFIG" ]; then
    echo "ERROR: unCLIP config not found at $UNCLIP_CONFIG"
    exit 1
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python -m infer.infer_EEG_fMRI_e2e \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --prior_timesteps $PRIOR_TIMESTEPS \
    --num_samples $NUM_SAMPLES \
    --save_blurry $SAVE_BLURRY \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --backbone $EEG_ENCODER_TYPE \
    --datasets_dir $DATASETS_DIR \
    --images_df_dir $IMAGES_DF_DIR \
    --normalize_fmri $NORMALIZE_FMRI \
    --emb_source $EMB_SOURCE \
    --atms_emb_size $ATMS_EMB_SIZE \
    --out_mlp_dim $OUT_MLP_DIM \
    --atms_drop_proj $ATMS_DROP_PROJ \
    --atms_d_model $ATMS_D_MODEL \
    --atms_n_heads $ATMS_N_HEADS \
    --atms_d_ff $ATMS_D_FF \
    --atms_dropout $ATMS_DROPOUT \
    --atms_factor $ATMS_FACTOR \
    --pooling_type $POOLING_TYPE \
    --embedding_dim $EMBEDDING_DIM \
    --mlp_layers $MLP_LAYERS \
    --attention_heads $ATTENTION_HEADS \
    --num_tokens $NUM_TOKENS \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --mse_scale $MSE_SCALE \
    --infonce_scale $INFONCE_SCALE \
    --proto_distill_scale $PROTO_DISTILL_SCALE \
    --temperature $TEMPERATURE"

# Add optional arguments
if [ -f "$UNCLIP_CKPT" ] && [ -f "$UNCLIP_CONFIG" ]; then
    CMD="$CMD --unclip_ckpt $UNCLIP_CKPT --unclip_config $UNCLIP_CONFIG"
fi

if [ "$SAVE_BLURRY" = "true" ] && [ -f "$AUTOENC_CKPT" ]; then
    CMD="$CMD --autoenc_ckpt $AUTOENC_CKPT"
fi

if [ -n "$MAX_BATCHES" ]; then
    CMD="$CMD --max_batches $MAX_BATCHES"
fi

# ── Execute ───────────────────────────────────────────────────────────────────
echo "Running command:"
echo "$CMD"
echo ""
eval $CMD
