# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brain2Image aligns EEG (THINGS-EEG dataset) and fMRI (NSD dataset) brain signals into a shared embedding space using paired images viewed by subjects in both datasets. The aligned EEG embeddings are then fed through MindEye2's generation pipeline to reconstruct images from EEG alone.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate BRAIN2IMAGE
pip install setuptools==81.0.0  # required for TensorBoard compatibility
```

If pandas import fails with libc++ errors, run `bash libc++FIX.sh` (fixes LD_LIBRARY_PATH).

Key dependencies: Python 3.12, PyTorch 2.6.0 (CUDA 12.6), mne, h5py, lmdb, einops, dalle2-pytorch, diffusers.

## Data Pipeline (run in order)

```bash
# 1. Repack raw datasets into DataFrames
python datasets/things-eeg_repack.py
python datasets/nsd-fmri_repack.py

# 2. Match THINGS and NSD images via CLIP embedding cosine similarity + K-means clustering
python preprocess/process_things_nsd_images_clustering.py

# 3. Build LMDB alignment dataset (downloads MindEye2 checkpoints automatically)
python preprocess/process_EEG_fMRI_align.py
```

Outputs land in `datasets/processed/`. The LMDB dataset is split 80/10/10 (train/val/test).

## Training

Three training tasks, each launched via bash scripts in `run_script/`:

```bash
# EEG-fMRI Alignment (primary task)
bash run_script/run_train_EEG_fMRI_align.sh

# EEG Classification (auxiliary task — 5-class K-means cluster prediction)
bash run_script/run_train_EEG_classify.sh

# EEG-to-Image End-to-End Generation (alignment + MindEye2 diffusion prior)
bash run_script/run_train_EEG_fMRI_e2e.sh
```

All scripts support switching between CBraMod and ATMS backbones via the `EEG_ENCODER_TYPE` variable. Dataset paths are auto-selected by backbone (CBraMod uses 200Hz data, ATMS uses 250Hz data).

Direct invocation (all training modules use `python -m`):
```bash
python -m train.train_EEG_fMRI_align \
    --datasets_dir datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01 \
    --foundation_dir datasets/processed/cbramod/pretrained_weights.pth \
    --use_pretrained_weights true \
    --epochs 100 --batch_size 64 --lr 1e-4

python -m train.train_EEG_fMRI_e2e \
    --datasets_dir datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01_250Hz \
    --foundation_dir datasets/processed/atms/sub-01.pth \
    --mindeye2_ckpt_path datasets/processed/mindeye2/sub-01_last_full.pth \
    --backbone ATMS --epochs 150 --batch_size 64 --lr 3e-4
```

Checkpoints and TensorBoard logs go to `runs/<task>_<timestamp>/`.

## Architecture

**Data flow:**
```
THINGS-EEG (63ch, 200Hz) ──┐
                            ├── paired images → LMDB dataset (with cluster labels)
NSD-fMRI → MindEye2 ───────┘   (fMRI projected to 4096-dim via ridge regression)
```

**Three models, layered:**

1. **EEG_fMRI_Align** (`model/EEG_fMRI_align.py`) — EEG encoder + projection → 4096-dim, aligned to fMRI via MSE + InfoNCE + prototypical distillation losses.

2. **EEG_Classify** (`model/EEG_classify.py`) — EEG encoder + MLP classifier → 5-class K-means cluster prediction.

3. **EEG_fMRI_E2E** (`model/EEG_fMRI_e2e.py`) — Wraps EEG_fMRI_Align + BrainNetwork + BrainDiffusionPrior. Full pipeline:
   ```
   EEG → EEG_fMRI_Align → 4096-dim → BrainNetwork (MLP Mixer) → (B, 256, 1664) CLIP tokens
                                     → BrainDiffusionPrior → diffusion loss
                                     + blurry branch → VAE latents (B, 4, 28, 28) + ConvNeXt features (B, 49, 512)
   ```

**EEG backbones** (two options):
- **CBraMod** (`model/CBraMod/cbramod.py`) — 12-layer criss-cross transformer. Input: `(B, 63, 1, 200)` → Output: `(B, 63, 1, 200)`. Requires a pooling head (`flatten`, `attention`, or `multitoken_vit`) to project to 4096-dim.
- **ATMS** (`model/encoders/atms_eeg_encoder.py`) — iTransformer + ShallowConvNet + projection MLP. Input: `(B, 63, 250)` → Output: `(B, 4096)` directly.

**MindEYE2 components** (`model/MindEYE2/`):
- `BrainNetwork` — MLP Mixer backbone (4096 → 256×1664 CLIP patch tokens + optional blurry branch)
- `BrainDiffusionPrior` — diffusion prior on CLIP patch tokens (100 timesteps)
- Utilities: `soft_clip_loss`, `soft_cont_loss`, `mixco`, `mixco_nce`, `cosine_anneal`

## Key Conventions

- **Package-style imports only** — run from project root with `python -m train.<module>`. No relative imports in training scripts.
- **MixCo augmentation** is applied to 4096-dim embeddings only, never to raw EEG signals. Original (unmixed) embeddings are kept for alignment loss; mixed embeddings go to the generation branch.
- **Optimizer LR groups** — EEG backbone gets 0.2× the base LR; alignment head, BrainNetwork, and diffusion prior get full LR. Weight decay is excluded from bias and LayerNorm parameters.
- **E2E warm-start** — the E2E model can load a pretrained alignment checkpoint via `--align_model_dir` before training the full pipeline.
- **Model config** is passed as a nested dict with keys `EEG_Encoder`, `Loss`, and (for E2E) `Generation`. No YAML/TOML config files; all configuration via argparse + bash scripts.
- **EEG normalization** — raw EEG values are divided by 100 in the dataset class. fMRI optionally unit-normalized.
- **Generation targets are pre-computed** — ViT-bigG CLIP tokens, SD VAE latents, and ConvNeXt features are stored in the LMDB dataset. No frozen vision models are loaded at training time.

## Key Files

| File | Purpose |
|------|---------|
| `model/EEG_fMRI_align.py` | Alignment model + loss computation |
| `model/EEG_fMRI_e2e.py` | End-to-end generation model (wraps align + MindEye2) |
| `model/EEG_classify.py` | Classification model (5-class) |
| `model/encoders/cbramod_eeg_encoder.py` | CBraMod encoder with pooling strategies |
| `model/encoders/atms_eeg_encoder.py` | ATMS encoder with attention + projection |
| `model/CBraMod/cbramod.py` | CBraMod backbone |
| `model/MindEYE2/` | BrainNetwork, BrainDiffusionPrior, loss utilities |
| `model/layers/` | Reusable layers: attention, attention_pooling, multi_token_ViT, fully_connected |
| `data/EEG_fMRI_align_dataset.py` | LMDB dataset for alignment/classification |
| `data/EEG_fMRI_generation_e2e_dataset.py` | Extended dataset with pre-computed generation targets |
| `train/train_EEG_fMRI_align.py` | Alignment training loop |
| `train/train_EEG_fMRI_e2e.py` | E2E generation training loop |
| `train/train_EEG_classify.py` | Classification training loop |
| `preprocess/process_EEG_fMRI_align.py` | Builds LMDB dataset from repacked data |
| `preprocess/process_things_nsd_images_clustering.py` | Image matching via CLIP + K-means |

## Notes

- No formal test suite; modules have `if __name__ == "__main__"` blocks for ad-hoc testing.
- No linting or formatting configuration is set up.
- `MindEyeV2/` is a separate git repo (cloned from MedARC-AI/MindEyeV2) used for fMRI ridge regression. It is gitignored from the main repo.
- Alignment training uses AdamW + CosineAnnealingLR; E2E training uses AdamW + OneCycleLR.
