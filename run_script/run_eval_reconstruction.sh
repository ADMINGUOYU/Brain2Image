#!/bin/bash
# Evaluate image reconstruction quality using MindEyeV2 metrics.
# Computes: PixCorr, SSIM, AlexNet(2), AlexNet(5), InceptionV3, CLIP, EffNet-B, SwAV
#
# Run from the project root:
#   bash run_script/run_eval_reconstruction.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR="runs/Mar04_EEGE2E/subj-01-wME2-ckpt_1772638509/inference-epoch100/images"
OUTPUT_DIR=""  # leave empty to save alongside images_dir (parent directory)

# ── Settings ──────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
BATCH_SIZE=40
USE_BLURRY_ENHANCEMENT="false"  # set to "true" to blend recon*0.75 + blurry*0.25

# ── Validation ────────────────────────────────────────────────────────────────
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: images directory not found: $IMAGES_DIR"
    exit 1
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python -m eval.evaluate_reconstruction \
    --images_dir $IMAGES_DIR \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --use_blurry_enhancement $USE_BLURRY_ENHANCEMENT"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

echo "Running: $CMD"
echo ""
eval $CMD
