# This is the dataset class for THINGS/NSD align task
# Please run 'preprocess/process_EEG_fMRI_align.py' to generate the dataset before using this code
# It yields:
#     'eeg': list of EEG data (batch_size, num_channels, num_timepoints)
#     'fmri': list of fMRI data (batch_size, embedding_dim)
#     'label': list of labels (image clustered category) (batch_size,)
#     'things_img_idx': list of image indices corresponding to each sample (batch_size,)
#     'nsd_img_idx': list of image indices corresponding to each sample (batch_size,)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import pickle

import typing


class EEG_fMRI_Align_Dataset(Dataset):

    """
    EEG_fMRI_Align_Dataset
    """

    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'train', 
                 normalize_fmri: bool = True):
        
        super(EEG_fMRI_Align_Dataset, self).__init__()

        self.db = lmdb.open(data_dir, 
                            readonly = True, 
                            lock = False, 
                            readahead = True, 
                            meminit = False)
        
        self.normalize_fmri = normalize_fmri

        with self.db.begin(write = False) as txn:
            # we only keep keys that belong to the current mode (train/val/test)
            self.keys = [key.decode() for key, _ in txn.cursor() if key.decode().startswith(mode + '_')]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Retrieve data from LMDB
        key = self.keys[idx]

        # Load data from LMDB
        with self.db.begin(write = False) as txn:
            data: dict = pickle.loads(txn.get(key.encode()))
        
        # Extract data
        EEG = data['eeg']  # EEG: (1, 63, 200) 
        fMRI = data['fmri']  # fMRI: (1, 4096,)
        label = data['label']
        things_img_idx = data['things_img_idx']
        nsd_img_idx = data['nsd_img_idx']

        # Assert shape
        assert EEG.shape == (1, 63, 200), f"Expected EEG shape (1, 63, 200), but got {EEG.shape}"
        assert fMRI.shape == (1, 4096), f"Expected fMRI shape (1, 4096), but got {fMRI.shape}"

        # Normalize EEG
        EEG = EEG / 100

        # Optionally normalize fMRI to unit norm
        if self.normalize_fmri:
            fMRI_norm = np.linalg.norm(fMRI)
            if fMRI_norm > 0:
                fMRI = fMRI / fMRI_norm

        return EEG, fMRI, label, things_img_idx, nsd_img_idx
    
    def collate(self, batch) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        EEG = np.array([x[0] for x in batch]).squeeze()
        fMRI = np.array([x[1] for x in batch]).squeeze()
        label = np.array([x[2] for x in batch])
        things_img_idx = np.array([x[3] for x in batch])
        nsd_img_idx = np.array([x[4] for x in batch])
        return torch.from_numpy(EEG).float(), torch.from_numpy(fMRI).float(), torch.from_numpy(label).long(), torch.from_numpy(things_img_idx).long(), torch.from_numpy(nsd_img_idx).long()

# helper function to get data loaders
def get_data_loader(datasets_dir: str, 
                    batch_size: int = 32,
                    normalize_fmri: bool = True)\
                    -> typing.Dict[str, DataLoader]:

    # create datasets
    train_set = EEG_fMRI_Align_Dataset(datasets_dir, mode = 'train', normalize_fmri = normalize_fmri)
    val_set = EEG_fMRI_Align_Dataset(datasets_dir, mode = 'val', normalize_fmri = normalize_fmri)
    test_set = EEG_fMRI_Align_Dataset(datasets_dir, mode = 'test', normalize_fmri = normalize_fmri)
    
    # verbose
    print(f"Dataset sizes: train = {len(train_set)}, val = {len(val_set)}, test = {len(test_set)}")
    print(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}")
    print(f"fMRI normalization: {'enabled' if normalize_fmri else 'disabled'}")
    
    # create data loaders
    data_loader = {
        'train': DataLoader(
            train_set,
            batch_size = batch_size,
            collate_fn = train_set.collate,
            shuffle=True,
        ),
        'val': DataLoader(
            val_set,
            batch_size = batch_size,
            collate_fn = val_set.collate,
            shuffle=False,
        ),
        'test': DataLoader(
            test_set,
            batch_size = batch_size,
            collate_fn = test_set.collate,
            shuffle=False,
        ),
    }
    return data_loader

# Test code
if __name__ == "__main__":
    # Example usage
    datasets_dir = 'path/to/dataset/dir'  # Update with actual path
    batch_size = 16
    data_loader = get_data_loader(datasets_dir, batch_size)

    # Iterate through one batch of training data
    for EEG_batch, fMRI_batch, label_batch, things_img_idx_batch, nsd_img_idx_batch in data_loader['train']:
        print(f"EEG batch shape: {EEG_batch.shape}")  # Expected: (batch_size, 63, 1, 200)
        print(f"fMRI batch shape: {fMRI_batch.shape}")  # Expected: (batch_size, 4096)
        print(f"label batch shape: {label_batch.shape}")
        print(f"things_img_idx batch shape: {things_img_idx_batch.shape}")
        print(f"nsd_img_idx batch shape: {nsd_img_idx_batch.shape}")
        break