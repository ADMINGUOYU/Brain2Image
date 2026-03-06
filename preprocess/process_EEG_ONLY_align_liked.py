# This script prepares an EEG-only dataset in alignment-dataset style.
# It mirrors process_EEG_fMRI_align.py but operates without any NSD/fMRI data.
#
# Each LMDB sample contains EEG data, an empty 'nsd_data' list, a K-means
# cluster label (derived from THINGS CLIP embeddings), and a THINGS image index.
# WARNING: we're not using the image specifically viewed by the selected subjects
#
# All subjects listed in 'things_subjects' are combined into ONE LMDB dataset.
#
# NOTE:
# meta_info = {
#     'eeg_take_mean':             eeg_take_mean,
#     'fmri_take_mean':            False,
#     'dataset_mix_mode':          'eeg_only',
#     'split_ratios':              split_ratios,
#     'split_seed':                split_seed,
#     'exclude_THINGS_test_split': filter_things_test_split,
#     'things_subjects':           things_subjects,
#     'nsd_subjects':              [],
#     'num_clusters':              num_clusters,
# }

# Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import lmdb
import pickle
import typing
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import signal

# --------------- Start of configuration --------------- #

# Path to the processed datasets directory
processed_dir = "datasets/processed"

# THINGS EEG subjects to include (list of subject IDs)
# All subjects are combined into a SINGLE LMDB dataset.
things_subjects = ['sub-01']

# Path to the image split info CSV produced by process_EEG_fMRI_align.py
# (optional). When provided, images that appeared in the paired alignment
# dataset keep their already-assigned split.
# align_split_info_csv = "datasets/processed/eeg_fmri_align_datasets/" + \
#                        "....csv"
# align_split_info_csv = None  # Set to None to disable CSV-guided splitting
align_split_info_csv = None

# EEG resampling target frequency
TARGET_FREQ = 250  # Hz (use 200 for CBraMod, 250 for ATMS)

# Whether to average EEG trials for the same image (reduces sample count to 1 per image)
eeg_take_mean = False

# Whether to exclude THINGS test-split images (test split has 80 trials vs 4 for train)
# Output directory name gets "_no_things_test" suffix when True to avoid confusion.
filter_things_test_split = True

# Number of K-means clusters used to label THINGS images
# Cluster labels are computed from CLIP embeddings of THINGS images.
num_clusters = 5

# Split ratios (must sum to 1.0)
split_ratios = {
    'train': 0.8,
    'val':   0.1,
    'test':  0.1,
}

# Random seed for reproducibility
split_seed = 42

# ---------------- End of configuration ---------------- #

# Validate split ratios
assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, \
    f"split_ratios must sum to 1.0, got {sum(split_ratios.values())}"

# Load THINGS images DataFrame (needed for CLIP embeddings + cluster labelling)
things_images_df_path = os.path.join(processed_dir, "things_images_df.pkl")
if not os.path.exists(things_images_df_path):
    raise FileNotFoundError(
        f"THINGS images DataFrame not found at {things_images_df_path}. "
        "Please run datasets/things-eeg_repack.py first."
    )
things_images_df: pd.DataFrame = pd.read_pickle(things_images_df_path)
required_cols = ['image_index', 'image_embedding', 'split']
missing = [c for c in required_cols if c not in things_images_df.columns]
if missing:
    raise ValueError(f"Missing columns in things_images_df: {missing}")
print(f"\033[92mLoaded things_images_df with {len(things_images_df)} rows.\033[0m")

# Optionally drop THINGS test-split images before clustering so that cluster
# labels are consistent with (and comparable to) the aligned dataset.
# NOTE: The lookup dict (image_index_to_cluster) is built from the same filtered
# DataFrame, so any test-split image that is absent here will be skipped
# later (logged as 'skipped') rather than crashing.
if filter_things_test_split:
    things_images_df_for_clustering = things_images_df[things_images_df['split'] != 'test']
else:
    things_images_df_for_clustering = things_images_df
print(f"Images available for clustering: {len(things_images_df_for_clustering)}")

# Compute K-means cluster labels on all THINGS image CLIP embeddings
# WARNING: we're not using the image specifically viewed by the selected subjects
print(f"Performing K-means clustering (k = {num_clusters}) on THINGS CLIP embeddings...")
all_embeddings = np.stack(things_images_df_for_clustering['image_embedding'].values)  # (N, embedding_dim)
kmeans = KMeans(n_clusters = num_clusters, random_state = split_seed, n_init = 'auto')
cluster_assignments = kmeans.fit_predict(all_embeddings)  # (N,)
print(f"\033[92mClustering done. Cluster distribution: "
      f"{ {c: int((cluster_assignments == c).sum()) for c in range(num_clusters)} }\033[0m")

# Build a lookup dict: things_image_index -> cluster_label
image_index_to_cluster: typing.Dict[int, int] = {
    int(row['image_index']): int(cluster_assignments[i])
    for i, (_, row) in enumerate(things_images_df_for_clustering.iterrows())
}

# Collect EEG data from ALL subjects into shared flat lists.
# A global sequential index (position in these lists) is used as the LMDB key,
# so there is no collision even when multiple subjects viewed the same image.
things_img_idx_list: typing.List[int] = []
label_list: typing.List[int] = []
eeg_data_list: typing.List[np.ndarray] = []

for things_subject in things_subjects:
    print(f"\n{'='*60}")
    print(f"Loading THINGS subject: {things_subject}")
    print(f"{'='*60}")

    # Locate the EEG data pickle for this subject
    things_eeg_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('things_eeg_data_df_') and file.endswith('.pkl') and things_subject in file:
            things_eeg_data_df_path = os.path.join(processed_dir, file)
            break
    if things_eeg_data_df_path is None:
        raise FileNotFoundError(
            f"Could not find things_eeg_data_df file for subject {things_subject} in {processed_dir}.\n"
            f"Please run datasets/things-eeg_repack.py on subject {things_subject} first."
        )

    things_eeg_data_df: pd.DataFrame = pd.read_pickle(things_eeg_data_df_path)
    things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
    print(f"\033[92mLoaded {len(things_eeg_data_df)} EEG rows for {things_subject}.\033[0m")

    # Optionally exclude test-split images
    if filter_things_test_split:
        things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['split'] != 'test']
        print(f"After filtering test split: {len(things_eeg_data_df)} rows.")

    # Get unique image indices for this subject
    unique_image_indices = things_eeg_data_df['image_index'].unique().tolist()
    print(f"Unique THINGS images for {things_subject}: {len(unique_image_indices)}")

    skipped = 0
    for img_idx in unique_image_indices:
        if img_idx not in image_index_to_cluster:
            # Image was not clustered (e.g., from a split excluded during clustering)
            skipped += 1
            continue
        eeg_row = things_eeg_data_df[things_eeg_data_df['image_index'] == img_idx]
        things_img_idx_list.append(img_idx)
        label_list.append(image_index_to_cluster[img_idx])
        eeg_data_list.append(eeg_row.iloc[0]['eeg'])  # (num_trials, channels, timepoints)

    if skipped > 0:
        print(f"\033[93mWarning: Skipped {skipped} images for {things_subject} "
              f"(no cluster label or EEG data found).\033[0m")
    del things_eeg_data_df

print(f"\n\033[92mTotal entries collected across all subjects: {len(eeg_data_list)} "
      f"(one entry per unique (subject, image) pair; trials stored inside each entry)\033[0m")

# Resample EEG to TARGET_FREQ
print(f"Resampling EEG to {TARGET_FREQ} Hz...")
for i in tqdm(range(len(eeg_data_list)), desc = "Resampling EEG"):
    eeg_data_list[i] = signal.resample(
        eeg_data_list[i], TARGET_FREQ, axis = 2
    ).astype(np.float32)
print(f"EEG shape after resampling: {eeg_data_list[0].shape}")

# ------------------------------------------------------------------ #
# Train / val / test split
#
#   1. If align_split_info_csv is set and the file exists, load it and
#      honour the pre-assigned splits for images that already appeared in
#      the paired (aligned) dataset.  Remaining images (EEG-only extras)
#      are distributed so that the OVERALL split ratios match split_ratios.
#   2. Otherwise (or if the file is not found), fall back to the original
#      strategy: overlapping images → train, rest split by ratio.
# ------------------------------------------------------------------ #

# Verbose
print(f"\n>>> Performing train/val/test split >>>")

# Small epsilon used to keep train_test_split fractions strictly inside (0, 1)
_SPLIT_EPS = 1e-9

# Build a lookup dict: things_image_index -> list of global indices in our dataset
things_img_idx_to_pair_indices: typing.Dict[int, typing.List[int]] = {}
for idx, t_idx in enumerate(things_img_idx_list):
    things_img_idx_to_pair_indices.setdefault(t_idx, []).append(idx)

# Initialize empty split lists
training_indices: typing.List[int] = []
val_indices:      typing.List[int] = []
test_indices:     typing.List[int] = []

# Attempt to load pre-assigned split info from the aligned dataset
align_split_df: typing.Optional[pd.DataFrame] = None
if align_split_info_csv is not None:
    if os.path.exists(align_split_info_csv):
        align_split_df = pd.read_csv(align_split_info_csv)
        required_cols = {'things_img_idx', 'split'}
        if not required_cols.issubset(align_split_df.columns):
            print(f"\033[91mWarning: align_split_info_csv at {align_split_info_csv} "
                  f"is missing columns {required_cols - set(align_split_df.columns)}. "
                  f"Falling back to default split.\033[0m")
            align_split_df = None
        else:
            print(f"\033[92mLoaded align split info CSV ({len(align_split_df)} rows) "
                  f"from {align_split_info_csv}\033[0m")
            # Pre-compute set for O(1) lookup
            align_split_set = set(align_split_df['things_img_idx'].values)
    else:
        print(f"\033[93mWarning: align_split_info_csv was specified as '{align_split_info_csv}' "
              f"but the file was not found. Falling back to default split.\033[0m")

# If we have valid align split info, assign samples accordingly and keep track of unassigned ones
if align_split_df is not None:

    # unassigned list
    unassigned_things_img_idx_to_pair_indices: typing.Dict[int, typing.List[int]] = {}

    # Assign samples to splits based on align_split_df
    for img_idx, pair_indices in things_img_idx_to_pair_indices.items():
        if img_idx in align_split_set:
            split = align_split_df[align_split_df['things_img_idx'] == img_idx]['split'].iloc[0]
            if split == 'train':
                training_indices.extend(pair_indices)
            elif split == 'val':
                val_indices.extend(pair_indices)
            elif split == 'test':
                test_indices.extend(pair_indices)
        else:
            # Image index not found in align_split_df, add to unassigned dict for later processing
            unassigned_things_img_idx_to_pair_indices[img_idx] = pair_indices
    
    # override the original dict with the unassigned one for later processing
    things_img_idx_to_pair_indices = unassigned_things_img_idx_to_pair_indices

    # Verbose
    print(f"\033[92mAssigned {len(training_indices)} samples to train, "
          f"{len(val_indices)} to val, {len(test_indices)} to test based on the CSV provided.\033[0m")

# Now we check for the unassigned samples, if there are any duplicates, assign them to train
overlap_indices: typing.Set[int] = set()
for indices in things_img_idx_to_pair_indices.values():
    if len(indices) > 1:
        overlap_indices.update(indices)
print(f"\033[93mFound {len(overlap_indices)} entries sharing an image index with another entry "
      f"(same image seen by multiple subjects or multiple times) — assigned to train.\033[0m")
# assign all overlapping samples to training set
training_indices.extend(overlap_indices)

# Calculate how many samples are left for splitting after assigning overlaps to training
assigned_indices = set(training_indices + val_indices + test_indices)
remaining_indices = [idx for idx in range(len(label_list)) if idx not in assigned_indices]

if len(remaining_indices) == 0:
    print("\033[91mWarning: All samples overlap. Assigning everything to training set.\033[0m")
else:
    # Note that we already have something in training set, val set and testing set (latter is
    # assigned based on the CSV if provided, otherwise empty). We want to split the remaining
    # samples according to split_ratios, but we need to adjust the ratios to account for the
    # already assigned samples.
    remaining_split_ratios = {
        'train': (split_ratios['train'] * len(label_list) - len(training_indices)) / len(remaining_indices) if len(remaining_indices) > 0 else 0,
        'val':   (split_ratios['val']   * len(label_list) - len(val_indices))      / len(remaining_indices) if len(remaining_indices) > 0 else 0,
        'test':  (split_ratios['test']  * len(label_list) - len(test_indices))     / len(remaining_indices) if len(remaining_indices) > 0 else 0,
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
            remaining_indices, test_size = val_test_size, random_state = split_seed
        )
        test_ratio = remaining_split_ratios['test'] / val_test_size
        if test_ratio <= 0 or test_ratio >= 1.0:
            val_indices.extend(temp_idx)
        else:
            val_idx, test_idx = train_test_split(
                temp_idx, test_size = test_ratio, random_state = split_seed
            )
            training_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

print(f"\033[92mSplit: train = {len(training_indices)}, val = {len(val_indices)}, test = {len(test_indices)}\033[0m")

# ------------------------------------------------------------------ #
# Initialise a single LMDB for all subjects combined
# ------------------------------------------------------------------ #
subjects_tag = "_".join(things_subjects)
test_suffix = "_no_things_test" if filter_things_test_split else ""
output_dir = (
    f"{processed_dir}/eeg_fmri_align_datasets/"
    f"things_{subjects_tag}_eeg_only{test_suffix}_{TARGET_FREQ}Hz"
)
if os.path.exists(output_dir):
    print(f"\033[93mOutput directory {output_dir} already exists. Removing it ...\033[0m")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"\n>>> Initializing LMDB database at {output_dir} >>>")
db = lmdb.open(output_dir, map_size=53687091200)

# Save metadata (same structure as alignment dataset, NSD fields are empty)
meta_info = {
    'eeg_take_mean':             eeg_take_mean,
    'fmri_take_mean':            False,          # no fMRI in this dataset
    'dataset_mix_mode':          'eeg_only',
    'split_ratios':              split_ratios,
    'split_seed':                split_seed,
    'exclude_THINGS_test_split': filter_things_test_split,
    'things_subjects':           things_subjects, # subjects combined into this dataset (one or many)
    'nsd_subjects':              [],              # empty — no NSD subject
    'num_clusters':              num_clusters,
}
txn = db.begin(write = True)
txn.put(key = b'meta_info', value = pickle.dumps(meta_info))
txn.commit()

# ------------------------------------------------------------------ #
# Write samples to LMDB
# Each sample dict follows the alignment-dataset schema exactly:
#   {
#       'eeg':            np.ndarray  (1, channels, timepoints)
#       'nsd_data':       []          (empty — no NSD subject)
#       'label':          int
#       'things_img_idx': int
#   }
# ------------------------------------------------------------------ #
for split_name, indices in zip(['train', 'val', 'test'],
                               [training_indices, val_indices, test_indices]):
    print(f"\n>>> Processing {split_name} split with {len(indices)} samples >>>")
    txn = db.begin(write=True)
    for idx in tqdm(indices, desc = f"Processing {split_name} samples"):
        eeg_data       = eeg_data_list[idx]   # (num_trials, channels, timepoints)
        label          = label_list[idx]
        things_img_idx = things_img_idx_list[idx]

        # Optionally average EEG trials
        if eeg_take_mean:
            eeg_data = np.mean(eeg_data, axis=0, keepdims = True)

        # One LMDB entry per EEG trial (mirrors cross-product expansion in alignment script)
        for trial_i in range(eeg_data.shape[0]):
            data_dict = {
                'eeg':            eeg_data[trial_i][np.newaxis, :],  # (1, channels, timepoints)
                'nsd_data':       [],                                 # empty — no NSD
                'label':          label,
                'things_img_idx': things_img_idx,
            }
            sample_key = f"{split_name}_{idx:05d}_{trial_i:05d}"
            txn.put(key = sample_key.encode(), value = pickle.dumps(data_dict))
    txn.commit()

db.close()
print(f"\nFinished processing. Dataset saved at {output_dir}.")

# Sanity check
print(f"\nSanity check for {output_dir} ...")
db = lmdb.open(output_dir, readonly = True, lock = False, readahead = True, meminit = False)
with db.begin(write = False) as txn:
    keys = [key.decode() for key, _ in txn.cursor()]
print(f"Total entries in LMDB: {len(keys) - 1} samples + 1 meta_info key.")
split_counts = {'train': 0, 'val': 0, 'test': 0}
for key in keys:
    for s in split_counts:
        if key.startswith(s + '_'):
            split_counts[s] += 1
print(f"Sample counts by split: {split_counts}")
# Verify a sample structure
train_keys = [k for k in keys if k.startswith('train_')]
if train_keys:
    with db.begin(write = False) as txn:
        sample = pickle.loads(txn.get(train_keys[0].encode()))
    print(f"Sample structure keys: {list(sample.keys())}")
    print(f"  EEG shape:      {sample['eeg'].shape}")
    print(f"  nsd_data:       {sample['nsd_data']}  (empty list — EEG-only dataset)")
    print(f"  label:          {sample['label']}")
    print(f"  things_img_idx: {sample['things_img_idx']}")
db.close()

# Verbose
print(f"\nDone processing.\n\033[92mDataset saved at: {output_dir}\033[0m")