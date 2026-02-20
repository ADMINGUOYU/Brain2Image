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
#     'fmri': list of fMRI data (num_samples, embedding_dim)
#     'label': list of labels (image clustered category)
#     'things_img_idx': list of image indices corresponding to each sample
#     'nsd_img_idx': list of image indices corresponding to each sample
# }
# 'val': { ... }
# 'test': { ... }

# Import necessary libraries
import os
import torch
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
# NOTE: in (things subject, nsd subject)
# NOTE: we LOOP every combination of subjects
subjects =  [('sub-01', 'sub-01')]

# EEG resampling parameters
TARGET_FREQ = 200

# Mean processing
eeg_take_mean = False
fmri_take_mean = True

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

# ---------------- End of configuration ---------------- #

# First we have to download the ckpt from MindEYE2
# TODO: safe to call if already downloaded, will return the path if so.
def download_ckpt(subject_number: int):
    # construct URL (make sure subject number in TWO digit)
    ckpt_url = f"https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj{subject_number:02d}_pretrained_40sess_24bs/last.pth?download=true"
    ckpt_path = processed_dir + f"/mindeye2/sub-{subject_number:02d}_last.pth"
    if not os.path.exists(ckpt_path):
        print(f"Downloading MindEYE2 checkpoint for subject {subject_number}...")
        # create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(ckpt_path), exist_ok = True)
        # use wget to download the file
        os.system(f"wget -O {ckpt_path} {ckpt_url}")
        # Opens the file and only keep ridge keys to save space
        checkpoint = torch.load(ckpt_path, map_location = 'cpu')
        ridge_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'ridge' in k}
        torch.save({'model_state_dict': ridge_state_dict}, ckpt_path)
        print(f"Checkpoint downloaded and saved to {ckpt_path}.")
    else:
        print(f"Checkpoint for subject {subject_number} already exists at {ckpt_path}.")
    return ckpt_path
# Now, we download
mindeye2_ckpts = []
for _, nsd_subject in subjects:
    mindeye2_ckpts.append(download_ckpt(int(nsd_subject[-2:])))

# Now we loop through the paired data and construct the dataset
for (things_subject, nsd_subject), mindeye2_ckpt in zip(subjects, mindeye2_ckpts):
    print(f"\nProcessing subject pair: Things {things_subject} and NSD {nsd_subject}")

    # Load paired data
    # in file: paired_images_subject_THINGS_<things_subj>_NSD_<nsd_subject>.csv
    # csv has 3 columns: things_image_index, nsd_image_index, cluster_label
    match_data_path = f"{processed_dir}/paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subject}.csv"
    if not os.path.exists(match_data_path):
        print(f"Paired data file not found at {match_data_path}. Make sure to run ./preprocess/process_things_nsd_images_clustering.py first to generate the paired data.")
        continue
    
    # Read CSV
    match_data = pd.read_csv(match_data_path)
    print(f"Loaded {len(match_data)} matched pairs from {match_data_path}.")

    # Load things EEG data and nsd fMRI embedding data from dataframes
    things_eeg_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('things_eeg_data_df_') and file.endswith('.pkl'):
            if things_subject in file:
                things_eeg_data_df_path = os.path.join(processed_dir, file)
                break
    if things_eeg_data_df_path is None:
        raise FileNotFoundError(f"Could not find the things_eeg_data_df file for subject {things_subject} in {processed_dir}.")
    nsd_fmri_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('nsd_fmri_data_df_') and file.endswith('.pkl'):
            if nsd_subject in file:
                nsd_fmri_data_df_path = os.path.join(processed_dir, file)
                break
    if nsd_fmri_data_df_path is None:
        raise FileNotFoundError(f"Could not find the nsd_fmri_data_df file for subject {nsd_subject} in {processed_dir}.")
    things_eeg_data_df = pd.read_pickle(things_eeg_data_df_path)
    things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
    nsd_fmri_data_df = pd.read_pickle(nsd_fmri_data_df_path)
    nsd_fmri_data_df = nsd_fmri_data_df[nsd_fmri_data_df['subject'] == nsd_subject]
    print(f"Loaded things EEG data with {len(things_eeg_data_df)} samples and NSD fMRI data with {len(nsd_fmri_data_df)} samples.")

    # Get index pairs and labels from match_data
    things_img_idx_list = match_data['things_image_index'].tolist()
    nsd_img_idx_list = match_data['nsd_image_index'].tolist()
    label_list = match_data['cluster_label'].tolist()
    # Now we fetch eeg and fmri data for each matched pair
    # based on things_image_index and nsd_image_index
    eeg_data_list = []
    fmri_data_list = []
    for things_img_idx, nsd_img_idx in zip(things_img_idx_list, nsd_img_idx_list):
        
        # Fetch EEG data for the things image index
        eeg_row = things_eeg_data_df[things_eeg_data_df['image_index'] == things_img_idx]
        if len(eeg_row) == 0:
            print(f"Warning: No EEG data found for Things image index {things_img_idx}. Skipping this pair.")
            continue
        elif len(eeg_row) > 1:
            print(f"Warning: Multiple EEG entries found for Things image index {things_img_idx}. Using the first one.")
        eeg_data = eeg_row.iloc[0]['eeg']  # shape (num_trails, num_timepoints, 250 Hz)

        # Fetch fMRI embedding data for the nsd image index
        fmri_row = nsd_fmri_data_df[nsd_fmri_data_df['image_index'] == nsd_img_idx]
        if len(fmri_row) == 0:
            print(f"Warning: No fMRI data found for NSD image index {nsd_img_idx}. Skipping this pair.")
            continue
        elif len(fmri_row) > 1:
            print(f"Warning: Multiple fMRI entries found for NSD image index {nsd_img_idx}. Using the first one.")
        fmri_data = fmri_row.iloc[0]['fmri']  # shape (num_trails, voxels dimension)

        # Append to lists
        eeg_data_list.append(eeg_data)
        fmri_data_list.append(fmri_data)

    # assert we've fetched data for all pairs
    assert len(eeg_data_list) == len(fmri_data_list) == len(label_list) == len(match_data), "Mismatch in number of samples fetched for EEG, fMRI, and labels."
    print(f"Successfully fetched EEG and fMRI data for {len(eeg_data_list)} matched pairs.")

    # Now we create the linear layer for fmri
    # First we load the ckpt and get the ridge regression parameters
    checkpoint = torch.load(mindeye2_ckpt, map_location = device)
    ridge_state_dict = checkpoint['model_state_dict']
    # Keys containing 'ridge': ['ridge.linears.0.weight', 'ridge.linears.0.bias']
    # we only needs these two parameters for the linear layer
    ridge_weight = ridge_state_dict['ridge.linears.0.weight']
    ridge_bias = ridge_state_dict['ridge.linears.0.bias']
    print(f"Loaded ridge regression parameters from checkpoint: weight shape {ridge_weight.shape}, bias shape {ridge_bias.shape}.")
    # We infer the input and output dimension from the weight shape
    input_dim = ridge_weight.shape[1]
    output_dim = ridge_weight.shape[0]
    print(f"Inferred ridge regression input dimension: {input_dim}, output dimension: {output_dim}.")
    # assert
    assert input_dim == fmri_data_list[0].shape[1], f"Input dimension of ridge regression ({input_dim}) does not match fMRI data dimension ({fmri_data_list[0].shape[1]})."
    # We create the torch linear layer and set its parameters
    projection_layer = torch.nn.Linear(input_dim, output_dim)
    with torch.no_grad():
        projection_layer.weight.copy_(ridge_weight)
        projection_layer.bias.copy_(ridge_bias)
        projection_layer.to(device)
        print("Created projection layer and loaded ridge regression parameters.")
    # We do the forward pass to get the hidden features for each fmri data
    fmri_data_hidden_list = []
    for fmri_data in tqdm(fmri_data_list, desc = "Processing fMRI data"):
        fmri_tensor = torch.from_numpy(fmri_data).float().to(device)
        with torch.no_grad():
            hidden_features = projection_layer(fmri_tensor).cpu().numpy()
        fmri_data_hidden_list.append(hidden_features)
    print(f"Processed fMRI data to hidden features with shape {fmri_data_hidden_list[0].shape}.")
    # Assign back to fmri_data_list
    del fmri_data_list  # free up memory
    fmri_data_list = fmri_data_hidden_list

    # We need to resample EEG data 250 Hz -> 200 Hz
    # use signal library
    for i in tqdm(range(len(eeg_data_list)), desc = "Resampling EEG data"):
        eeg_data = eeg_data_list[i]  # shape (num_trails, num_timepoints, 250 Hz)
        # Resample
        resampled_eeg = signal.resample(eeg_data, TARGET_FREQ, axis = 2)
        eeg_data_list[i] = resampled_eeg.astype(np.float32)  # shape (num_trails, new_timepoints, 200 Hz)
    print(f"Resampled EEG data to target frequency {TARGET_FREQ} Hz. New shape: {eeg_data_list[0].shape}.")

    # now we have all lists ready: eeg_data_list, fmri_data_list, label_list, things_img_idx_list, nsd_img_idx_list
    assert len(eeg_data_list) == len(fmri_data_list) == len(label_list) == len(things_img_idx_list) == len(nsd_img_idx_list), "Mismatch in number of samples across data lists."
    # we have to do the splitting
    training_indices = []
    val_indices = []
    test_indices = []
    # NOTE: make sure we don't put same image in different splits
    #       (A, 1010) and (A, 648) is not allowed
    #       (A, 648) and (C, 648) is not allowed
    #       note that there might be overlap in things_img_idx_list and nsd_img_idx_list
    # Let's find these cases first
    things_img_idx_to_pair_indices = {}
    nsd_img_idx_to_pair_indices = {}
    for idx, (things_idx, nsd_idx) in enumerate(zip(things_img_idx_list, nsd_img_idx_list)):
        if things_idx not in things_img_idx_to_pair_indices:
            things_img_idx_to_pair_indices[things_idx] = []
        things_img_idx_to_pair_indices[things_idx].append(idx)
        if nsd_idx not in nsd_img_idx_to_pair_indices:
            nsd_img_idx_to_pair_indices[nsd_idx] = []
        nsd_img_idx_to_pair_indices[nsd_idx].append(idx)
    # Now we find the indices that have overlap
    overlap_indices = set()
    for indices in things_img_idx_to_pair_indices.values():
        if len(indices) > 1:
            overlap_indices.update(indices)
    for indices in nsd_img_idx_to_pair_indices.values():
        if len(indices) > 1:
            overlap_indices.update(indices)
    print(f"Found {len(overlap_indices)} overlapping pairs that share the same image index.")
    # We put these to training set
    training_indices.extend(overlap_indices)
    # Now we get the list of remaining indices
    remaining_indices = [idx for idx in range(len(label_list)) if idx not in overlap_indices]
    # calculate corrected split ratios (since we assigned something)
    # training should be smaller regarding remaining_indices
    remaining_split_ratios = {
        'train': (split_ratios['train'] * len(label_list) - len(overlap_indices)) / len(remaining_indices),
        'val': split_ratios['val'] * len(label_list) / len(remaining_indices),
        'test': split_ratios['test'] * len(label_list) / len(remaining_indices)
    }
    # Now we split the remaining indices
    train_idx, temp_idx = train_test_split(remaining_indices, test_size = remaining_split_ratios['val'] + remaining_split_ratios['test'], random_state = split_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size = remaining_split_ratios['test'] / (remaining_split_ratios['val'] + remaining_split_ratios['test']), random_state = split_seed)
    training_indices.extend(train_idx)
    val_indices.extend(val_idx)
    test_indices.extend(test_idx)
    print(f"Split data into {len(training_indices)} training samples, {len(val_indices)} validation samples, and {len(test_indices)} test samples.")

    # We init lmdb
    output_dir = f"{processed_dir}/eeg_fmri_align_datasets/things_{things_subject}_nsd_{nsd_subject}"
    os.makedirs(output_dir, exist_ok = True)
    # Remove output_dir if already exists to avoid confusion
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Removing it ...")
        os.system(f"rm -rf {output_dir}")
    print(f"Initializing LMDB database at {output_dir}...")
    db = lmdb.open(output_dir, map_size = 53687091200)
    # Save take mean / dataset mix mode / split ratio / split seed
    # to lmdb as meta info for future reference
    meta_info = {
        'eeg_take_mean': eeg_take_mean,
        'fmri_take_mean': fmri_take_mean,
        'dataset_mix_mode': dataset_mix_mode,
        'split_ratios': split_ratios,
        'split_seed': split_seed
    }
    txn = db.begin(write = True)
    txn.put(key = 'meta_info'.encode(), value = pickle.dumps(meta_info))
    txn.commit()
    # We loop training / testing / validation indices
    for split_name, indices in zip(['train', 'val', 'test'], [training_indices, val_indices, test_indices]):
        print(f"Processing {split_name} split with {len(indices)} samples...")
        for idx in tqdm(indices, desc = f"Processing {split_name} samples"):
            # fetch data
            eeg_data = eeg_data_list[idx]  # shape (num_trails, new_timepoints, 200 Hz)
            fmri_data = fmri_data_list[idx]  # shape (num_trails, hidden_dim)
            label = label_list[idx]
            things_img_idx = things_img_idx_list[idx]
            nsd_img_idx = nsd_img_idx_list[idx]
            # Perform mean processing if needed
            if eeg_take_mean:
                eeg_data = np.mean(eeg_data, axis = 0, keepdims = True)  # shape (1, time_point, 200 Hz)
            if fmri_take_mean:
                fmri_data = np.mean(fmri_data, axis = 0, keepdims = True)  # shape (1, hidden_dim)
            # perform dataset mixing
            data_dictionaries = []
            if dataset_mix_mode == 'cross':
                # For each EEG sample, we find all matching FMRI samples
                for eeg_idx in range(eeg_data.shape[0]):
                    for fmri_idx in range(fmri_data.shape[0]):
                        # we keep original shape
                        data_dict = {
                            'eeg': eeg_data[eeg_idx][ np.newaxis, : ],     # shape (1, time_point, 200 Hz)
                            'fmri': fmri_data[fmri_idx][ np.newaxis, : ],  # shape (1, hidden_dim)
                            'label': label,
                            'things_img_idx': things_img_idx,
                            'nsd_img_idx': nsd_img_idx
                        }
                        data_dictionaries.append(data_dict)
            else:
                # If not cross, just use the original data
                # Place-holder for now
                data_dict = {
                    'eeg': eeg_data,   # shape (num_trails, time_point, 200 Hz)
                    'fmri': fmri_data, # shape (num_trails, hidden_dim)
                    'label': label,
                    'things_img_idx': things_img_idx,
                    'nsd_img_idx': nsd_img_idx
                }
                data_dictionaries.append(data_dict)
            # Save each mixed sample to LMDB
            for i, data_dict in enumerate(data_dictionaries):
                sample_key = f"{split_name}_{idx:05d}_{i:05d}"
                txn = db.begin(write = True)
                txn.put(key = sample_key.encode(), value = pickle.dumps(data_dict))
                txn.commit()
    # After we processed all samples, we close the database
    db.close()
    print(f"Finished processing subject pair: Things {things_subject} and NSD {nsd_subject}. Dataset saved at {output_dir}.")

    # Sanity check:
    print(f"\nSanity check for the created LMDB dataset at {output_dir}...")
    db = lmdb.open(output_dir, 
                   readonly = True, 
                   lock = False, 
                   readahead = True, 
                   meminit = False)
    with db.begin(write = False) as txn:
        keys = [key.decode() for key, _ in txn.cursor()]
    print(f"Total samples in LMDB: {len(keys) - 1} (excluding meta_info). Sample keys: {keys[:5]}...")
    # We count the number of samples in each split
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for key in keys:
        if key.startswith('train_'):
            split_counts['train'] += 1
        elif key.startswith('val_'):
            split_counts['val'] += 1
        elif key.startswith('test_'):
            split_counts['test'] += 1
    print(f"Sample counts by split: {split_counts}")
    db.close()