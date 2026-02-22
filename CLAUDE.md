# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brain2Image aligns EEG (THINGS-EEG dataset) and fMRI (NSD dataset) brain signals into a shared embedding space using paired images viewed by subjects in both datasets. The goal is to enable cross-modal translation between fast/portable EEG and high-resolution fMRI representations.

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

Two training tasks are available:

**EEG-fMRI Alignment** (primary task):
```bash
bash run_script/run_train_EEG_fMRI_align.sh
```

**EEG Classification** (auxiliary task — 5-class K-means cluster prediction):
```bash
bash run_script/run_train_EEG_classify.sh
```

Both scripts support switching between CBraMod and ATMS backbones via the `EEG_ENCODER_TYPE` variable. Dataset paths are auto-selected by backbone (CBraMod uses 200Hz data, ATMS uses 250Hz data).

Direct invocation example:
```bash
python -m train.train_EEG_fMRI_align \
    --datasets_dir datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01 \
    --foundation_dir datasets/processed/cbramod/pretrained_weights.pth \
    --use_pretrained_weights true \
    --epochs 100 --batch_size 64 --lr 1e-4
```

Checkpoints and TensorBoard logs go to `runs/EEG_fMRI_align_<timestamp>/`.

## Architecture

**Data flow:**
```
THINGS-EEG (63ch, 200Hz) ──┐
                            ├── paired images → LMDB dataset (with cluster labels)
NSD-fMRI → MindEye2 ───────┘   (fMRI projected to 4096-dim via ridge regression)
```

**Model pipeline** (`model/EEG_fMRI_align.py`):
1. **EEG backbone** — two options:
   - **CBraMod** (`model/CBraMod/cbramod.py`) — pretrained EEG foundation model; 12-layer criss-cross transformer with spectral patch embedding. Input: `(B, 63, 1, 200)` → Output: `(B, 63, 1, 200)`.
   - **ATMS** (`model/encoders/atms_eeg_encoder.py`) — attention-based time series model with projection. Input: `(B, 63, 250)` → Output: `(B, 4096)`.
2. **Pooling head** (`model/encoders/cbramod_eeg_encoder.py`, CBraMod only) — three strategies: `flatten` (MLP), `attention` (dynamic pooling + MLP), `multitoken_vit` (learnable tokens + transformer). Output: `(B, 4096)`.
3. **Alignment module** — multi-head attention where fMRI is the query and EEG is key/value, producing a `(B, 1, 4096)` aligned embedding.

**Loss functions** (all three combined with configurable scales):
- MSE — direct reconstruction
- InfoNCE — contrastive alignment
- Prototypical distillation — cluster-level alignment using K-means labels from preprocessing

**Optimizer**: AdamW with two learning rate groups — lower LR for the CBraMod backbone, higher for the alignment head. Cosine annealing scheduler, gradient clipping applied.

## Key Files

| File | Purpose |
|------|---------|
| `model/EEG_fMRI_align.py` | Top-level alignment model + loss computation |
| `model/EEG_classify.py` | Classification model (5-class K-means clusters) |
| `model/encoders/cbramod_eeg_encoder.py` | CBraMod EEG encoder with pooling strategies |
| `model/encoders/atms_eeg_encoder.py` | ATMS EEG encoder with attention + projection |
| `model/CBraMod/cbramod.py` | CBraMod backbone (pretrained EEG foundation model) |
| `model/layers/` | Reusable layers: attention, attention_pooling, multi_token_ViT, fully_connected |
| `data/EEG_fMRI_align_dataset.py` | LMDB dataset; EEG normalized by /100, optional fMRI unit norm |
| `data/EEG_fMRI_generation_e2e_dataset.py` | Extended dataset with CLIP embeddings and raw images for generation |
| `train/train_EEG_fMRI_align.py` | Alignment training loop, validation metrics (MSE, cosine sim, retrieval accuracy) |
| `train/train_EEG_classify.py` | Classification training loop |
| `preprocess/process_EEG_fMRI_align.py` | Builds the LMDB dataset from repacked data |
| `preprocess/process_things_nsd_images_clustering.py` | Image matching via CLIP + K-means clustering |
| `datasets/things-eeg_repack.py` | THINGS-EEG raw → DataFrame + CLIP embeddings |
| `datasets/nsd-fmri_repack.py` | NSD-fMRI raw → DataFrame + CLIP embeddings |

## Notes

- All training modules are invoked as `python -m train.<module>` (package-style imports).
- No formal test suite exists; modules have `if __name__ == "__main__"` blocks for ad-hoc testing.
- No linting or formatting configuration is set up.
- `MindEyeV2/` is a separate git repo (cloned from MedARC-AI/MindEyeV2) used for fMRI ridge regression and the generation pipeline. It is gitignored from the main repo.
