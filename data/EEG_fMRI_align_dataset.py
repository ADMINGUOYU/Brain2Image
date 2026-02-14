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
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx) -> typing.Tuple[np.ndarray, np.ndarray]:

        # Retrieve data from LMDB
        key = self.keys[idx]

        # Load the EEG-fMRI pair
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        
        # Extract EEG and fMRI
        EEG = pair['EEG']  # EEG: (63, 1, 200)
        fMRI = pair['fMRI']  # fMRI: (4096,)

        # Normalize EEG
        EEG = EEG / 100

        # Optionally normalize fMRI to unit norm
        if self.normalize_fmri:
            fMRI_norm = np.linalg.norm(fMRI)
            if fMRI_norm > 0:
                fMRI = fMRI / fMRI_norm

        return EEG, fMRI
    
    def collate(self, batch) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        EEG = np.array([x[0] for x in batch])
        fMRI = np.array([x[1] for x in batch])
        return torch.from_numpy(EEG).float(), torch.from_numpy(fMRI).float()

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
    datasets_dir = 'path/to/dataset'  # Update with actual path
    batch_size = 16
    data_loader = get_data_loader(datasets_dir, batch_size)

    # Iterate through one batch of training data
    for EEG_batch, fMRI_batch in data_loader['train']:
        print(f"EEG batch shape: {EEG_batch.shape}")  # Expected: (batch_size, 63, 1, 200)
        print(f"fMRI batch shape: {fMRI_batch.shape}")  # Expected: (batch_size, 4096)
        break