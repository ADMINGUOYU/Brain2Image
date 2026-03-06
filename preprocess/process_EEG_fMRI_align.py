# This script will prepare the dataset for EEG-fMRI align model
# We'll use the paired images to get their corresponding data
# and use these data to construct dataset
# Note that the same image is NOT allowed to be in the same split
# NOTE: (A, B) and (A, C) is not allowed - we put these into training set.

# We download MindEYE2 ckpt and extract model.ridge parameters
# the model.ridge type is defined as:
# class RidgeRegression(torch.nn.Module):
#    # make sure to add weight_decay when initializing optimizer to enable regularization
#    def __init__(self, input_sizes, out_features): 
#        super(RidgeRegression, self).__init__()
#        self.out_features = out_features
#        self.linears = torch.nn.ModuleList([
#                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
#            ])
#    def forward(self, x, subj_idx):
#        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
#        return out
# NOTE: it's expected to output 4096 hidden features

# NOTE: the model ckpt can be downloaded in this formatted URL
# https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj<nsd_subject>_pretrained_40sess_24bs/last.pth?download=true

# The dataset will be saved in lmdb format, with the following keys:
# 'train': {
#     'eeg': list of EEG data (num_samples, num_channels, num_timepoints)
#     'nsd_data': list of dicts, one per NSD subject:
#         [{'subject': str, 'fmri': (1, 4096), 'nsd_img_idx': int}, ...]
#     'label': list of labels (image clustered category)
#     'things_img_idx': list of image indices corresponding to each sample
# }
# 'val': { ... }
# 'test': { ... }

# Import necessary libraries
import os
import torch
import typing
import numpy as np
import pandas as pd
import lmdb
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import signal

# --------------- Start of configuration --------------- #

# Paths to necessary files
processed_dir = "datasets/processed"

# The subjects we want to use 
# NOTE: please make sure you've processed the paired data
#       using ./preprocess/process_things_nsd_images_clustering.py
#       OR   ./preprocess/process_things_nsd_images_multi_subject_clustering.py
# NOTE: we LOOP every combination of subjects
# Format: (things_subject, nsd_subject_or_tuple)
#   Single NSD subject:   ('sub-01', 'sub-01')
#   Multiple NSD subjects: ('sub-01', ('sub-01', 'sub-02', 'sub-05', 'sub-07'))
subjects =  [('sub-01', ('sub-01', 'sub-02', 'sub-05', 'sub-07'))]

# EEG resampling parameters
TARGET_FREQ = 250 # for ATM
# TARGET_FREQ = 200 # for CBraMod

# Mean processing
eeg_take_mean = False
fmri_take_mean = True   # NOTE: we take subject-wise mean (NOT) cross mean

# If we need to load filtered THINGS test split's CSV
# NOTE: if True, we will look for the CSV file with "_no_things_test" suffix
#       and we will also save the processed dataset with "_no_things_test" suffix to avoid confusion
filter_things_test_split = True

# Dataset mix mode
# NOTE: Mean processing will reduce number of samples to 1 !!!
# -> 'cross': if EEG has 4 samples per image, fMRI has 3 samples per image
#             cross will generate 3 x 4 = 12 samples per image pair
dataset_mix_mode = 'cross'

# Split ratios for train, val, test
split_ratios = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}
# Split seed for reproducibility
split_seed = 42

# Device for processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Make a copy of full ckpt? (downloaded from MindEYE2)
# You might need this if you wish to load the full ckpt in end-to-end model
# saved under: <processed_di>/mindeye2/sub-<subject_number>_last_full.pth
preserve_full_ckpt = True

# Hugging Face mirror (optional, set to False to disable)
MIRROR = False
mirror = 'https://hf-mirror.com/'

# ---------------- End of configuration ---------------- #

# First we have to download the ckpt from MindEYE2
# TODO: safe to call if already downloaded, will return the path if so.
def download_ckpt(subject_number: int):
    # construct URL (make sure subject number in TWO digit)
    ckpt_url = f"https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj{subject_number:02d}_pretrained_40sess_24bs/last.pth?download=true"
    # set mirror url
    if MIRROR:
        # replace the Hugging Face URL with the mirror link
        ckpt_url = ckpt_url.replace("https://huggingface.co/", mirror)
    # construct local path to save the ckpt
    ckpt_path = processed_dir + f"/mindeye2/sub-{subject_number:02d}_last.pth"
    if not os.path.exists(ckpt_path):
        print(f"Downloading MindEYE2 checkpoint for subject {subject_number}...")
        # create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(ckpt_path), exist_ok = True)
        # use wget to download the file
        os.system(f"wget -O {ckpt_path} {ckpt_url}")
        # If specified
        if preserve_full_ckpt:
            # we make a copy of the full ckpt
            full_ckpt_path = processed_dir + f"/mindeye2/sub-{subject_number:02d}_last_full.pth"
            os.system(f"cp {ckpt_path} {full_ckpt_path}")
        # Opens the file and only keep ridge keys to save space
        checkpoint = torch.load(ckpt_path, map_location = 'cpu')
        ridge_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'ridge' in k}
        torch.save({'model_state_dict': ridge_state_dict}, ckpt_path)
        print(f"Checkpoint downloaded and saved to {ckpt_path}.")
    else:
        print(f"Checkpoint for subject {subject_number} already exists at {ckpt_path}.")
    return ckpt_path


# Normalize subjects list to ensure all NSD subjects are in tuple form for uniform handling
subjects = [(things, nsd) if isinstance(nsd, tuple) else (things, (nsd,)) for things, nsd in subjects]
# We sort the NSD subjects in each tuple to ensure consistent ordering (important for naming and pairing)
subjects = [(things, tuple(sorted(nsd))) for things, nsd in subjects]

# Now, we download checkpoints for all NSD subjects
mindeye2_ckpts = []  # list of lists: one list of ckpt paths per subject config
for _, nsd_subject in subjects:
    ckpt_list = []
    for ns in nsd_subject:
        ckpt_list.append(download_ckpt(int(ns[-2:])))
    mindeye2_ckpts.append(ckpt_list)

# Now we loop through the paired data and construct the dataset
for (things_subject, nsd_subject_tuple), mindeye2_ckpt_list in zip(subjects, mindeye2_ckpts):

    # Prepare NSD subject tag for naming
    nsd_subjects_tag = "_".join(nsd_subject_tuple)
    print(f"\nProcessing subject pair: Things {things_subject} and NSD {nsd_subject_tuple}")

    # Load paired data CSV
    # For multi-subject, the CSV was produced by process_things_nsd_images_multi_subject_clustering.py
    # For single subject, the CSV was produced by process_things_nsd_images_clustering.py
    test_suffix = "_no_things_test" if filter_things_test_split else ""
    match_data_path = f"{processed_dir}/paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subjects_tag}{test_suffix}.csv"
    if not os.path.exists(match_data_path):
        raise FileNotFoundError(f"Paired data file not found at {match_data_path}. Make sure to run the pairing script first.")

    # Read CSV
    match_data = pd.read_csv(match_data_path)
    print(f"\033[92mLoaded {len(match_data)} matched pairs from {match_data_path}.\033[0m")

    # Get lists of THINGS image indices and labels
    things_img_idx_list = match_data['things_image_index'].tolist()
    label_list = match_data['cluster_label'].tolist()

    # Build a dict: nsd_subject -> list of nsd image indices (aligned with things_img_idx_list)
    # NOTE: compatible with old single subject CSV
    nsd_img_idx_lists: typing.Dict[str, typing.List[int]] = {}  # nsd_subject -> [nsd_img_idx, ...]
    if 'nsd_image_index' in match_data.columns:
        # Old single-subject CSV format
        nsd_img_idx_lists[nsd_subject_tuple[0]] = match_data['nsd_image_index'].tolist()
    else:
        # New multi-subject CSV format: columns like nsd_sub-01_image_index, nsd_sub-02_image_index, ...
        for ns in nsd_subject_tuple:
            col = f'nsd_{ns}_image_index'
            if col not in match_data.columns:
                raise ValueError(f"Expected column '{col}' in CSV but not found. Columns: {list(match_data.columns)}")
            nsd_img_idx_lists[ns] = match_data[col].tolist()

    # Load THINGS EEG data (RAW)
    things_eeg_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('things_eeg_data_df_') and file.endswith('.pkl'):
            if things_subject in file:
                things_eeg_data_df_path = os.path.join(processed_dir, file)
                break
    if things_eeg_data_df_path is None:
        raise FileNotFoundError(f"Could not find the things_eeg_data_df file for subject {things_subject} in {processed_dir}.")
    things_eeg_data_df = pd.read_pickle(things_eeg_data_df_path)
    things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
    print(f"\033[92mLoaded things EEG data with {len(things_eeg_data_df)} samples.\033[0m")

    # Load fMRI data and apply ridge regression for each NSD subject
    # Result: per_subject_fmri[nsd_subject] = list of hidden arrays (one per matched pair)
    per_subject_fmri: typing.Dict[str, typing.List[np.ndarray]] = {}  # nsd_subject -> list of np.ndarray (num_trials, 4096)
    # assert
    assert len(nsd_subject_tuple) == len(mindeye2_ckpt_list), \
        f"Number of NSD subjects in tuple ({len(nsd_subject_tuple)}) does not match number of downloaded ckpt lists ({len(mindeye2_ckpt_list)})."
    # LOOP for each NSD subject
    for ns, mindeye2_ckpt in zip(nsd_subject_tuple, mindeye2_ckpt_list):
        print(f"\n>>> Loading fMRI data for NSD {ns} >>>")
        nsd_fmri_data_df_path = None
        for file in os.listdir(processed_dir):
            if file.startswith('nsd_fmri_data_df_') and file.endswith('.pkl'):
                if ns in file:
                    nsd_fmri_data_df_path = os.path.join(processed_dir, file)
                    break
        if nsd_fmri_data_df_path is None:
            raise FileNotFoundError(f"Could not find the nsd_fmri_data_df file for subject {ns} in {processed_dir}.")
        nsd_fmri_data_df = pd.read_pickle(nsd_fmri_data_df_path)
        nsd_fmri_data_df = nsd_fmri_data_df[nsd_fmri_data_df['subject'] == ns]
        print(f"\033[92mLoaded NSD {ns}: {len(nsd_fmri_data_df)} fMRI samples.\033[0m")

        # Fetch fMRI data for each matched pair
        fmri_data_raw: typing.List[np.ndarray] = []
        nsd_idx_list = nsd_img_idx_lists[ns]
        for nsd_img_idx in nsd_idx_list:
            fmri_row = nsd_fmri_data_df[nsd_fmri_data_df['image_index'] == nsd_img_idx]
            if len(fmri_row) == 0:
                raise ValueError(f"No fMRI data found for NSD {ns} image index {nsd_img_idx}.")
            fmri_data_raw.append(fmri_row.iloc[0]['fmri'])  # (num_trials, voxels)
        del nsd_fmri_data_df

        # Load ridge regression checkpoint and project fMRI -> 4096-dim
        checkpoint = torch.load(mindeye2_ckpt, map_location = device)
        ridge_state_dict = checkpoint['model_state_dict']
        ridge_weight = ridge_state_dict['ridge.linears.0.weight']
        ridge_bias = ridge_state_dict['ridge.linears.0.bias']
        input_dim = ridge_weight.shape[1]
        output_dim = ridge_weight.shape[0]
        assert input_dim == fmri_data_raw[0].shape[1], \
            f"Ridge input dim ({input_dim}) != fMRI dim ({fmri_data_raw[0].shape[1]}) for NSD {ns}."
        projection_layer = torch.nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            projection_layer.weight.copy_(ridge_weight)
            projection_layer.bias.copy_(ridge_bias)
            projection_layer.to(device)
        print(f"Ridge regression ({ns}): {input_dim} -> {output_dim}")

        fmri_hidden_list: typing.List[np.ndarray] = []
        for fmri_data in tqdm(fmri_data_raw, desc = f"Projecting fMRI for NSD {ns}"):
            fmri_tensor = torch.from_numpy(fmri_data).float().to(device)
            with torch.no_grad():
                hidden = projection_layer(fmri_tensor).cpu().numpy()
            fmri_hidden_list.append(hidden)
        per_subject_fmri[ns] = fmri_hidden_list
        print(f"\033[92mProjected fMRI hidden shape for NSD ({ns}): {fmri_hidden_list[0].shape}\033[0m")
        del fmri_data_raw

    # Fetch EEG data for each matched pair
    eeg_data_list: typing.List[np.ndarray] = []
    for things_img_idx in things_img_idx_list:
        eeg_row = things_eeg_data_df[things_eeg_data_df['image_index'] == things_img_idx]
        if len(eeg_row) == 0:
            raise ValueError(f"No EEG data found for Things image index {things_img_idx}.")
        eeg_data_list.append(eeg_row.iloc[0]['eeg'])  # (num_trials, channels, timepoints)
    del things_eeg_data_df
    assert len(eeg_data_list) == len(things_img_idx_list)
    print(f"\033[92m\nSuccessfully fetched EEG data for {len(eeg_data_list)} matched pairs.\033[0m")

    # Resample EEG
    print(f"\n>>> Resampling EEG data to {TARGET_FREQ} Hz >>>")
    for i in tqdm(range(len(eeg_data_list)), desc = "Resampling EEG data"):
        eeg_data_list[i] = signal.resample(eeg_data_list[i], TARGET_FREQ, axis = 2).astype(np.float32)
    print(f"Resampled EEG to {TARGET_FREQ} Hz. Shape: {eeg_data_list[0].shape}")

    # Train / val / test splitting
    # Make sure the same image does not appear in different splits
    training_indices = []
    val_indices = []
    test_indices = []

    # Collect all image indices across all NSD subjects
    things_img_idx_to_pair_indices: typing.Dict[int, typing.List[int]] = {}
    all_nsd_img_idx_to_pair_indices: typing.Dict[int, typing.List[int]] = {}
    for idx, things_idx in enumerate(things_img_idx_list):
        things_img_idx_to_pair_indices.setdefault(things_idx, []).append(idx)
    for ns in nsd_subject_tuple:
        for idx, nsd_idx in enumerate(nsd_img_idx_lists[ns]):
            all_nsd_img_idx_to_pair_indices.setdefault(nsd_idx, []).append(idx)

    overlap_indices = set()
    for indices in things_img_idx_to_pair_indices.values():
        if len(indices) > 1:
            overlap_indices.update(indices)
    for indices in all_nsd_img_idx_to_pair_indices.values():
        if len(indices) > 1:
            overlap_indices.update(indices)
    print(f"\033[93mFound {len(overlap_indices)} overlapping pairs (assigned to train).\033[0m")
    training_indices.extend(overlap_indices)

    remaining_indices = [idx for idx in range(len(label_list)) if idx not in overlap_indices]

    if len(remaining_indices) == 0:
        # All pairs overlap; put everything in train, nothing to split
        print(f"\033[91mWarning: All pairs overlap. Assigning everything to training set.\033[0m")
        val_indices = []
        test_indices = []
    else:
        remaining_split_ratios = {
            'train': max(0, (split_ratios['train'] * len(label_list) - len(overlap_indices)) / len(remaining_indices)),
            'val': split_ratios['val'] * len(label_list) / len(remaining_indices),
            'test': split_ratios['test'] * len(label_list) / len(remaining_indices)
        }
        val_test_size = remaining_split_ratios['val'] + remaining_split_ratios['test']
        if val_test_size <= 0 or val_test_size >= 1.0:
            # Cannot split meaningfully; put all remaining in train
            training_indices.extend(remaining_indices)
        else:
            train_idx, temp_idx = train_test_split(remaining_indices,
                                                   test_size = val_test_size,
                                                   random_state = split_seed)
            test_ratio = remaining_split_ratios['test'] / val_test_size
            if test_ratio <= 0 or test_ratio >= 1.0:
                val_indices.extend(temp_idx)
            else:
                val_idx, test_idx = train_test_split(temp_idx,
                                                     test_size = test_ratio,
                                                     random_state = split_seed)
                training_indices.extend(train_idx)
                val_indices.extend(val_idx)
                test_indices.extend(test_idx)
    print(f"\033[92mSplit: train = {len(training_indices)}, val = {len(val_indices)}, test = {len(test_indices)}\033[0m")

    # Set output path for our dataset
    output_dir = f"{processed_dir}/eeg_fmri_align_datasets/things_{things_subject}_nsd_{nsd_subjects_tag}{test_suffix}_{TARGET_FREQ}Hz"

    # Save image split info CSV alongside the LMDB directory so that
    # process_EEG_ONLY_align_liked.py can honour the same split assignments.
    # CSV columns: things_img_idx, split
    split_info_csv_path = f"{output_dir}_things_image_split_info.csv"
    os.makedirs(os.path.dirname(split_info_csv_path), exist_ok = True)
    split_rows = []
    for idx in training_indices:
        split_rows.append({'things_img_idx': things_img_idx_list[idx], 'split': 'train'})
    for idx in val_indices:
        split_rows.append({'things_img_idx': things_img_idx_list[idx], 'split': 'val'})
    for idx in test_indices:
        split_rows.append({'things_img_idx': things_img_idx_list[idx], 'split': 'test'})
    pd.DataFrame(split_rows).to_csv(split_info_csv_path, index=False)
    print(f"\033[92mSaved image split info CSV to {split_info_csv_path}\033[0m")

    # Initialise LMDB
    if os.path.exists(output_dir):
        print(f"\033[93mOutput directory {output_dir} already exists. Removing it ...\033[0m")
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok = True)
    print(f"\n >>> Initializing LMDB database at {output_dir} >>>")
    db = lmdb.open(output_dir, map_size = 53687091200)

    # Save metadata (including NSD subject info)
    meta_info = {
        'eeg_take_mean': eeg_take_mean,
        'fmri_take_mean': fmri_take_mean,
        'dataset_mix_mode': dataset_mix_mode,
        'split_ratios': split_ratios,
        'split_seed': split_seed,
        'exclude_THINGS_test_split': filter_things_test_split,
        'things_subject': things_subject,
        'nsd_subjects': list(nsd_subject_tuple),
    }
    txn = db.begin(write = True)
    txn.put(key = 'meta_info'.encode(), value = pickle.dumps(meta_info))
    txn.commit()

    # Write samples to LMDB
    for split_name, indices in zip(['train', 'val', 'test'], [training_indices, val_indices, test_indices]):
        print(f"\n >>> Processing {split_name} split with {len(indices)} samples >>>")
        for idx in tqdm(indices, desc = f"Processing {split_name} samples"):
            eeg_data = eeg_data_list[idx]  # (num_trials, channels, timepoints)
            label = label_list[idx]
            things_img_idx = things_img_idx_list[idx]

            # Perform mean processing if needed
            if eeg_take_mean:
                eeg_data = np.mean(eeg_data, axis = 0, keepdims = True)

            # Build per-subject fMRI data and optionally apply mean
            nsd_subject_fmri_list = []
            for ns in nsd_subject_tuple:
                fmri_data = per_subject_fmri[ns][idx]  # (num_trials, 4096)
                nsd_img_idx = nsd_img_idx_lists[ns][idx]
                if fmri_take_mean:
                    fmri_data = np.mean(fmri_data, axis = 0, keepdims = True)
                nsd_subject_fmri_list.append({
                    'subject': ns,
                    'fmri': fmri_data,  # (num_trials or 1, 4096)
                    'nsd_img_idx': nsd_img_idx,
                })

            # Dataset mixing
            data_dictionaries = []
            if dataset_mix_mode == 'cross':
                # Cross product of EEG trials × first NSD subject's fMRI trials
                first_fmri = nsd_subject_fmri_list[0]['fmri']
                for eeg_idx in range(eeg_data.shape[0]):
                    for fmri_idx in range(first_fmri.shape[0]):
                        # Build nsd_data list for this sample
                        nsd_data = []
                        for s_i, subj_entry in enumerate(nsd_subject_fmri_list):
                            if s_i == 0:
                                fmri_trial = subj_entry['fmri'][fmri_idx]
                            else:
                                # Cycle through this subject's available trials.
                                # NOTE: if subjects have different trial counts, this wraps
                                # around so every cross-product entry gets a valid trial.
                                fmri_trial = subj_entry['fmri'][fmri_idx % subj_entry['fmri'].shape[0]]
                            nsd_data.append({
                                'subject': subj_entry['subject'],
                                'fmri': fmri_trial[np.newaxis, :],  # (1, 4096)
                                'nsd_img_idx': subj_entry['nsd_img_idx'],
                            })
                        data_dict = {
                            'eeg': eeg_data[eeg_idx][np.newaxis, :],  # (1, channels, timepoints)
                            'nsd_data': nsd_data,
                            'label': label,
                            'things_img_idx': things_img_idx,
                        }
                        data_dictionaries.append(data_dict)
            else:
                # No cross-product, store all trials
                nsd_data = []
                for subj_entry in nsd_subject_fmri_list:
                    nsd_data.append({
                        'subject': subj_entry['subject'],
                        'fmri': subj_entry['fmri'],
                        'nsd_img_idx': subj_entry['nsd_img_idx'],
                    })
                data_dict = {
                    'eeg': eeg_data,
                    'nsd_data': nsd_data,
                    'label': label,
                    'things_img_idx': things_img_idx,
                }
                data_dictionaries.append(data_dict)

            # Save each mixed sample to LMDB
            for i, data_dict in enumerate(data_dictionaries):
                sample_key = f"{split_name}_{idx:05d}_{i:05d}"
                txn = db.begin(write = True)
                txn.put(key = sample_key.encode(), value = pickle.dumps(data_dict))
                txn.commit()

    db.close()
    print(f"\nFinished processing. Dataset saved at {output_dir}.")

    # Sanity check
    print(f"\nSanity check for the created LMDB dataset at {output_dir}...")
    db = lmdb.open(output_dir, readonly = True, lock = False, readahead = True, meminit = False)
    with db.begin(write = False) as txn:
        keys = [key.decode() for key, _ in txn.cursor()]
    print(f"Total samples in LMDB: {len(keys) - 1} (excluding meta_info). Sample keys: {keys[:5]}...")
    # We count the number of samples in each split
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for key in keys:
        for s in split_counts:
            if key.startswith(s + '_'):
                split_counts[s] += 1
    print(f"Sample counts by split: {split_counts}")
    # Verify a sample structure
    with db.begin(write = False) as txn:
        sample_key = [k for k in keys if k.startswith('train_')][0]
        sample = pickle.loads(txn.get(sample_key.encode()))
        print(f"Sample structure keys: {list(sample.keys())}")
        print(f"  EEG shape: {sample['eeg'].shape}")
        print(f"  nsd_data: {len(sample['nsd_data'])} subject(s)")
        for nd in sample['nsd_data']:
            print(f"    subject = {nd['subject']}, fmri shape = {nd['fmri'].shape}, nsd_img_idx = {nd['nsd_img_idx']}")
    db.close()