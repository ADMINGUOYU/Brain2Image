# This script prepares the dataset for EEG-CLIP alignment model.
# It builds an LMDB containing EEG signals paired with ViT-H-14 CLIP CLS embeddings,
# bypassing fMRI entirely.
#
# The dataset will be saved in LMDB format with entries:
#   {
#     'eeg': np.ndarray,                   # (1, 63, TARGET_FREQ) float32
#     'clip_H_cls_embedding': np.ndarray,  # (1024,) float32
#     'things_img_idx': int,               # THINGS image index
#   }
#
# Run from project root:
#   python -m preprocess.process_EEG_CLIP_align

import os
import numpy as np
import pandas as pd
import lmdb
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import signal

# --------------- Start of configuration --------------- #

processed_dir = "datasets/processed"

# Subject(s) to process (THINGS EEG subjects)
subjects = ['sub-08']

# EEG resampling target frequency
# Use 250 for ATMS backbone, 200 for CBraMod backbone
# TARGET_FREQ = 250  # for ATMS
TARGET_FREQ = 200  # for CBraMod

# Whether to average EEG trials (reduces samples to 1 per image)
eeg_take_mean = False

# If True, load the CSV with "_no_things_test" suffix (excludes THINGS test split images)
filter_things_test_split = True

# Split ratios for train / val / test
split_ratios = {
    'train': 0.8,
    'val':   0.1,
    'test':  0.1,
}
split_seed = 42

# ---------------- End of configuration ---------------- #


for things_subject in subjects:

    print(f"\nProcessing THINGS subject: {things_subject}")

    # ------------------------------------------------------------------
    # Load THINGS images DataFrame to get available image indices
    # ------------------------------------------------------------------
    things_images_df_path = os.path.join(processed_dir, "things_images_df.pkl")
    if not os.path.exists(things_images_df_path):
        raise FileNotFoundError(
            f"THINGS images DataFrame not found at {things_images_df_path}. "
            "Please run datasets/things-eeg_repack.py first."
        )
    things_images_df = pd.read_pickle(things_images_df_path)
    required_cols = ['image_index', 'split']
    missing = [c for c in required_cols if c not in things_images_df.columns]
    if missing:
        raise ValueError(f"Missing columns in things_images_df: {missing}")
    print(f"\033[92mLoaded things_images_df with {len(things_images_df)} rows.\033[0m")

    # Build set of available image indices (optionally excluding test split)
    if filter_things_test_split:
        available_images = set(things_images_df[things_images_df['split'] != 'test']['image_index'].values)
    else:
        available_images = set(things_images_df['image_index'].values)
    print(f"Available images for processing: {len(available_images)}")

    # ------------------------------------------------------------------
    # Load THINGS EEG data
    # ------------------------------------------------------------------
    things_eeg_data_df_path = None
    for fname in os.listdir(processed_dir):
        if (fname.startswith('things_eeg_data_df_') and fname.endswith('.pkl')
                and things_subject in fname):
            things_eeg_data_df_path = os.path.join(processed_dir, fname)
            break
    if things_eeg_data_df_path is None:
        raise FileNotFoundError(
            f"Could not find things_eeg_data_df file for {things_subject} in {processed_dir}."
        )

    things_eeg_data_df = pd.read_pickle(things_eeg_data_df_path)
    things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
    print(f"\033[92mLoaded THINGS EEG: {len(things_eeg_data_df)} samples.\033[0m")

    # ------------------------------------------------------------------
    # Load CLIP-H CLS embeddings from things_images_emb.pkl
    # ------------------------------------------------------------------
    things_emb_path = os.path.join(processed_dir, 'things_images_emb.pkl')
    if not os.path.exists(things_emb_path):
        raise FileNotFoundError(
            f"Could not find {things_emb_path}. "
            "Run 'python -m datasets.emb_generation --input datasets/processed/things_images_df.pkl' first."
        )
    things_emb_df = pd.read_pickle(things_emb_path)
    if 'clip_H_cls_embedding' not in things_emb_df.columns:
        raise ValueError(
            "clip_H_cls_embedding column not found in things_images_emb.pkl. "
            "Re-run datasets/emb_generation.py to regenerate with ViT-H-14 support."
        )
    # Build lookup: image_index -> clip_H_cls_embedding (1024,)
    clip_H_lookup = {
        int(row['image_index']): row['clip_H_cls_embedding'].astype(np.float32)
        for _, row in things_emb_df.iterrows()
    }
    print(f"\033[92mLoaded CLIP-H embeddings for {len(clip_H_lookup)} images.\033[0m")

    # ------------------------------------------------------------------
    # Fetch EEG for each available image
    # ------------------------------------------------------------------
    things_img_idx_list = []
    eeg_data_list = []

    # Optionally exclude test-split images from EEG data
    if filter_things_test_split:
        things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['split'] != 'test']
        print(f"After filtering test split: {len(things_eeg_data_df)} EEG rows.")

    unique_image_indices = things_eeg_data_df['image_index'].unique().tolist()
    print(f"Unique THINGS images for {things_subject}: {len(unique_image_indices)}")

    skipped = 0
    for img_idx in unique_image_indices:
        if img_idx not in available_images:
            # Image not in available set (e.g., test split filtered out)
            skipped += 1
            continue
        eeg_row = things_eeg_data_df[things_eeg_data_df['image_index'] == img_idx]
        things_img_idx_list.append(img_idx)
        eeg_data_list.append(eeg_row.iloc[0]['eeg'])  # (num_trials, channels, timepoints)

    if skipped > 0:
        print(f"\033[93mWarning: Skipped {skipped} images for {things_subject} "
              f"(not in available images set).\033[0m")
    del things_eeg_data_df
    print(f"\033[92mFetched EEG for {len(eeg_data_list)} images.\033[0m")

    # ------------------------------------------------------------------
    # Resample EEG to TARGET_FREQ
    # ------------------------------------------------------------------
    print(f"\n>>> Resampling EEG to {TARGET_FREQ} Hz >>>")
    for i in tqdm(range(len(eeg_data_list)), desc="Resampling EEG"):
        eeg_data_list[i] = signal.resample(
            eeg_data_list[i], TARGET_FREQ, axis=2
        ).astype(np.float32)
    print(f"Resampled EEG shape: {eeg_data_list[0].shape}")

    # ------------------------------------------------------------------
    # Train / val / test split  (image-unique: same image stays in one split)
    # ------------------------------------------------------------------
    training_indices = []
    val_indices = []
    test_indices = []

    things_img_idx_to_pair_indices = {}
    for idx, things_idx in enumerate(things_img_idx_list):
        things_img_idx_to_pair_indices.setdefault(things_idx, []).append(idx)

    # Pairs with the same THINGS image go to training (overlap set)
    overlap_indices = set()
    for indices in things_img_idx_to_pair_indices.values():
        if len(indices) > 1:
            overlap_indices.update(indices)
    print(f"\033[93mFound {len(overlap_indices)} overlapping pairs (assigned to train).\033[0m")
    training_indices.extend(overlap_indices)

    remaining_indices = [idx for idx in range(len(things_img_idx_list)) if idx not in overlap_indices]

    if len(remaining_indices) == 0:
        print("\033[91mWarning: All pairs overlap. Assigning everything to training set.\033[0m")
    else:
        # Note that we already have something in training set (overlap samples).
        # We want to split the remaining samples according to split_ratios, but we need to
        # adjust the ratios to account for the already assigned samples.
        remaining_split_ratios = {
            'train': (split_ratios['train'] * len(things_img_idx_list) - len(training_indices)) / len(remaining_indices) if len(remaining_indices) > 0 else 0,
            'val':   (split_ratios['val']   * len(things_img_idx_list) - len(val_indices))      / len(remaining_indices) if len(remaining_indices) > 0 else 0,
            'test':  (split_ratios['test']  * len(things_img_idx_list) - len(test_indices))     / len(remaining_indices) if len(remaining_indices) > 0 else 0,
        }
        # Validate and clamp negative ratios
        for key in remaining_split_ratios:
            remaining_split_ratios[key] = max(0, remaining_split_ratios[key])
        # Renormalize if sum > 1
        ratio_sum = sum(remaining_split_ratios.values())
        if ratio_sum > 1:
            for key in remaining_split_ratios:
                remaining_split_ratios[key] /= ratio_sum
        val_test_size = remaining_split_ratios['val'] + remaining_split_ratios['test']
        if val_test_size <= 0 or val_test_size >= 1.0:
            training_indices.extend(remaining_indices)
        else:
            train_idx, temp_idx = train_test_split(
                remaining_indices, test_size=val_test_size, random_state=split_seed
            )
            test_ratio = remaining_split_ratios['test'] / val_test_size
            if test_ratio <= 0 or test_ratio >= 1.0:
                val_indices.extend(temp_idx)
            else:
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=test_ratio, random_state=split_seed
                )
                training_indices.extend(train_idx)
                val_indices.extend(val_idx)
                test_indices.extend(test_idx)

    print(f"\033[92mSplit: train={len(training_indices)}, val={len(val_indices)}, test={len(test_indices)}\033[0m")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    test_suffix = "_no_things_test" if filter_things_test_split else ""
    freq_suffix = f"_{TARGET_FREQ}Hz" if TARGET_FREQ != 200 else ""
    output_dir = (
        f"{processed_dir}/eeg_clip_align_datasets/"
        f"things_{things_subject}_clip{test_suffix}{freq_suffix}"
    )

    if os.path.exists(output_dir):
        print(f"\033[93mOutput directory {output_dir} already exists. Removing it...\033[0m")
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n>>> Initializing LMDB at {output_dir} >>>")

    db = lmdb.open(output_dir, map_size=10_737_418_240)  # 10 GB

    # Save metadata
    meta_info = {
        'eeg_take_mean': eeg_take_mean,
        'target_freq': TARGET_FREQ,
        'split_ratios': split_ratios,
        'split_seed': split_seed,
        'exclude_THINGS_test_split': filter_things_test_split,
        'things_subject': things_subject,
        'clip_model': 'ViT-H-14',
        'clip_embedding_dim': 1024,
    }
    txn = db.begin(write=True)
    txn.put(b'meta_info', pickle.dumps(meta_info))
    txn.commit()

    # ------------------------------------------------------------------
    # Write samples
    # ------------------------------------------------------------------
    global_counter = {}  # split -> running count for unique keys

    for split_name, indices in zip(['train', 'val', 'test'],
                                    [training_indices, val_indices, test_indices]):
        print(f"\n>>> Processing {split_name} split ({len(indices)} pairs) >>>")
        sample_num = 0
        for idx in tqdm(indices, desc=f"Writing {split_name}"):
            eeg_data = eeg_data_list[idx]      # (num_trials, 63, TARGET_FREQ)
            things_img_idx = things_img_idx_list[idx]

            # Look up CLIP-H embedding
            if things_img_idx not in clip_H_lookup:
                raise ValueError(f"No CLIP-H embedding found for image index {things_img_idx}.")
            clip_H_emb = clip_H_lookup[things_img_idx]  # (1024,)

            # Optionally take mean over EEG trials
            if eeg_take_mean:
                eeg_data = np.mean(eeg_data, axis=0, keepdims=True)

            # One LMDB entry per EEG trial
            for trial_i in range(eeg_data.shape[0]):
                data_dict = {
                    'eeg': eeg_data[trial_i][np.newaxis, :].astype(np.float32),  # (1, 63, TARGET_FREQ)
                    'clip_H_cls_embedding': clip_H_emb,                           # (1024,)
                    'things_img_idx': int(things_img_idx),
                }
                sample_key = f"{split_name}_{idx:05d}_{trial_i:05d}"
                txn = db.begin(write=True)
                txn.put(sample_key.encode(), pickle.dumps(data_dict))
                txn.commit()
                sample_num += 1

        print(f"  Wrote {sample_num} samples for {split_name}.")

    db.close()
    print(f"\nFinished. Dataset saved at {output_dir}.")

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    print(f"\nSanity check for LMDB at {output_dir}...")
    db = lmdb.open(output_dir, readonly=True, lock=False, readahead=True, meminit=False)
    with db.begin(write=False) as txn:
        keys = [key.decode() for key, _ in txn.cursor()]
    print(f"Total entries: {len(keys) - 1} samples + 1 meta_info")
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for key in keys:
        for s in split_counts:
            if key.startswith(s + '_'):
                split_counts[s] += 1
    print(f"Counts by split: {split_counts}")
    # Verify one sample
    with db.begin(write=False) as txn:
        train_keys = [k for k in keys if k.startswith('train_')]
        if train_keys:
            sample = pickle.loads(txn.get(train_keys[0].encode()))
            print(f"Sample keys: {list(sample.keys())}")
            print(f"  EEG shape:              {sample['eeg'].shape}")
            print(f"  clip_H_cls_embedding:   {sample['clip_H_cls_embedding'].shape}")
            print(f"  things_img_idx:         {sample['things_img_idx']}")
    db.close()
