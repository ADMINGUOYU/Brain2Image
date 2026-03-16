"""
EEG-VAR Generation Dataset

Dataset for VAR stage 2 training.
Loads EEG from LMDB + raw images from HDF5 on-the-fly.

CRITICAL: HDF5 file is opened lazily in __getitem__ to avoid crashes with num_workers > 0.
Sharing file handles across processes causes segfaults.
"""

import os
import lmdb
import pickle
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple


class EEG_VAR_Generation_Dataset(Dataset):
    """
    Dataset for VAR stage 2 training.
    Loads EEG from LMDB + raw images from HDF5 on-the-fly.
    """

    def __init__(
        self,
        lmdb_dir: str,
        h5_path: str,
        mode: str = 'train',
        image_size: int = 256
    ):
        """
        Args:
            lmdb_dir: Path to LMDB directory containing EEG data
            h5_path: Path to HDF5 file containing raw images
            mode: 'train', 'val', or 'test'
            image_size: Target image size (default: 256)
        """
        self.lmdb_dir = lmdb_dir
        self.h5_path = h5_path
        self.mode = mode
        self.image_size = image_size

        # Load LMDB environment
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)

        # Load keys for the specified split
        self.keys = self._load_keys(mode)
        print(f"Loaded {len(self.keys)} samples for {mode} split")

        # Store HDF5 path (NOT the file handle)
        # File handle will be opened lazily in __getitem__
        self.h5_file = None

        # Image transforms: resize to 256×256, normalize to [-1, 1]
        # Match AVDE's augmentation strategy
        mid_reso = round(1.125 * image_size)  # 288 for training
        self.train_transform = transforms.Compose([
            transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def _load_keys(self, mode: str) -> list:
        """Load LMDB keys for the specified split."""
        # Use the same key schema as EEG_fMRI_Align_Dataset
        # Keys are prefixed with mode: train_0, train_1, val_0, etc.
        prefix = (mode + "_").encode()
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, _ in cursor.iternext():
                    k = key.decode()
                    if not k.startswith(mode + "_"):
                        break
                    keys.append(key)
        return keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            eeg: EEG tensor (63, 1, 200/250)
            image: Image tensor (3, 256, 256) in [-1, 1]
            things_img_idx: THINGS image index
        """
        # 1. Load EEG from LMDB
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))
        # EEG is stored as (1, 63, 200/250), reshape to (63, 1, 200/250)
        eeg = torch.from_numpy(data['eeg']).float().transpose(0, 1) / 100.0
        things_img_idx = data['things_img_idx']

        # 2. Lazy-open HDF5 file (per-worker, avoids multiprocessing crashes)
        if not hasattr(self, 'h5_file') or self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # 3. Load raw image from HDF5
        img_np = self.h5_file['images'][things_img_idx]  # (224, 224, 3) uint8
        image = Image.fromarray(img_np)

        # 4. Apply transforms
        if self.mode == 'train':
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        return eeg, image, things_img_idx

    def __del__(self):
        """Close HDF5 file handle when dataset is destroyed."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()
