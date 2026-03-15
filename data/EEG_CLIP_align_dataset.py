# Dataset for EEG-CLIP alignment task.
# Run preprocess/process_EEG_CLIP_align.py to generate the LMDB before use.
# Returns: (EEG, clip_H_embedding, things_img_idx)
#   EEG:               (B, 63, 1, T) float tensor
#   clip_H_embedding:  (B, 1024) float tensor
#   things_img_idx:    (B,) long tensor

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import pickle
import typing


class EEG_CLIP_Align_Dataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 mode: str = 'train',
                 normalize_clip: bool = True):

        super(EEG_CLIP_Align_Dataset, self).__init__()

        self.data_dir = data_dir
        self.db = None  # opened lazily in worker processes (LMDB is NOT fork-safe)
        self.normalize_clip = normalize_clip

        # Temporarily open LMDB to enumerate keys for this split
        _db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
        prefix = (mode + "_").encode()
        with _db.begin(write=False) as txn:
            cursor = txn.cursor()
            self.keys = []
            if cursor.set_range(prefix):
                for key, _ in cursor.iternext():
                    k = key.decode()
                    if not k.startswith(mode + "_"):
                        break
                    self.keys.append(k)
        _db.close()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx) -> typing.Tuple[np.ndarray, np.ndarray, int]:

        key = self.keys[idx]

        if self.db is None:
            self.db = lmdb.open(
                self.data_dir, readonly=True, lock=False, readahead=False, meminit=False
            )

        with self.db.begin(write=False) as txn:
            data: dict = pickle.loads(txn.get(key.encode()))

        EEG = data['eeg']                          # (1, 63, T)
        clip_H = data['clip_H_cls_embedding']      # (1024,)
        things_img_idx = data['things_img_idx']

        assert EEG.shape[0] == 1 and EEG.shape[1] == 63, \
            f"Unexpected EEG shape: {EEG.shape}"
        assert clip_H.shape == (1024,), \
            f"Unexpected clip_H shape: {clip_H.shape}"

        # Normalize EEG (matches fMRI pipeline convention)
        EEG = EEG / 100.0

        # Optionally L2-normalize CLIP embedding
        if self.normalize_clip:
            norm = np.linalg.norm(clip_H)
            if norm > 0:
                clip_H = clip_H / norm

        return EEG, clip_H, things_img_idx

    def collate(self, batch: typing.List[typing.Tuple[np.ndarray, np.ndarray, int]]) \
            -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        EEG = np.array([x[0] for x in batch]).squeeze(axis=1)  # (B, 63, T)
        clip_H = np.array([x[1] for x in batch])               # (B, 1024)
        things_img_idx = np.array([x[2] for x in batch])

        # Reshape EEG to (B, 63, 1, T) to match backbone input convention
        EEG = EEG[:, :, np.newaxis, :]  # (B, 63, 1, T)

        return (
            torch.from_numpy(EEG).float(),
            torch.from_numpy(clip_H).float(),
            torch.from_numpy(things_img_idx).long(),
        )


def get_clip_align_data_loader(datasets_dir: str,
                                batch_size: int = 32,
                                normalize_clip: bool = True,
                                num_workers: int = 4) \
        -> typing.Dict[str, DataLoader]:

    train_set = EEG_CLIP_Align_Dataset(datasets_dir, mode='train', normalize_clip=normalize_clip)
    val_set   = EEG_CLIP_Align_Dataset(datasets_dir, mode='val',   normalize_clip=normalize_clip)
    test_set  = EEG_CLIP_Align_Dataset(datasets_dir, mode='test',  normalize_clip=normalize_clip)

    print(f"Dataset sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"CLIP normalization: {'enabled' if normalize_clip else 'disabled'}")

    return {
        'train': DataLoader(
            train_set, batch_size=batch_size, collate_fn=train_set.collate,
            shuffle=True, num_workers=num_workers,
        ),
        'val': DataLoader(
            val_set, batch_size=batch_size, collate_fn=val_set.collate,
            shuffle=False, num_workers=num_workers,
        ),
        'test': DataLoader(
            test_set, batch_size=batch_size, collate_fn=test_set.collate,
            shuffle=False, num_workers=num_workers,
        ),
    }


if __name__ == "__main__":
    datasets_dir = 'datasets/processed/eeg_clip_align_datasets/things_sub-01_clip_no_things_test_250Hz'
    loaders = get_clip_align_data_loader(datasets_dir, batch_size=16)

    for EEG_batch, clip_H_batch, idx_batch in loaders['train']:
        print(f"EEG shape:    {EEG_batch.shape}")    # (B, 63, 1, 250)
        print(f"CLIP-H shape: {clip_H_batch.shape}") # (B, 1024)
        print(f"idx shape:    {idx_batch.shape}")
        break
