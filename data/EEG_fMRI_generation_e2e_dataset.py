# Dataset class for EEG-fMRI generation E2E task
# Extends the alignment dataset to also provide:
# - CLIP image embeddings (NO need ???)
# - raw images (for blurry reconstruction target).
# - Pre-computed generation targets: ViT-bigG CLIP tokens, VAE latents, ConvNeXt features
#
# NOTE: please run preprocess/process_EEG_fMRI_align.py
#       first to create the aligned EEG-fMRI LMDB datasets.
# NOTE: pkl data is loaded once in get_generation_data_loader() and shared
#       across all splits via _shared_data to avoid 3× memory duplication.
#
# Yields:
#   'eeg':                   (batch_size, 63, 1, 200 / 250)
#   'fmri':                  (batch_size, 4096)
#   'label':                 (batch_size,)
#   'things_img_idx':        (batch_size,)
#   'nsd_img_idx':           (batch_size,)
#   'things_clip_target':    (batch_size, clip_embed_dim)
#   'nsd_clip_target':       (batch_size, clip_embed_dim)
#   'things_image':          (batch_size, 3, 224, 224)
#   'nsd_image':             (batch_size, 3, 224, 224)
#   'clip_target_bigG':      (batch_size, 256, 1664)   — pre-computed ViT-bigG/14 patch tokens
#   'vae_latents':           (batch_size, 4, 28, 28)   — pre-computed SD VAE latents
#   'cnx_features':          (batch_size, 49, 512)     — pre-computed ConvNeXt features


# --------------- Start of configuration --------------- #

default_images_df_path = "datasets/processed"

# ---------------- End of configuration ---------------- #


# Import necessary libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import lmdb
import pickle

import typing

class EEG_fMRI_Generation_E2E_Dataset(Dataset):

    """
    Dataset EEG-fMRI generation end-to-end (E2E) pipeline.
    Loads EEG/fMRI from LMDB and image/CLIP data from repacked DataFrames.
    """

    def __init__(
        self,
        data_dir: str,
        images_df_dir: str = default_images_df_path,
        mode: str = "train",
        normalize_fmri: bool = True,
        load_images: bool = False,
        image_size: int = 224,
        emb_source: str = "things",
        _shared_data: typing.Optional[dict] = None,
    ):
        super().__init__()

        """
        Args:
            data_dir: Path to the LMDB dataset directory containing aligned EEG-fMRI data.
            images_df_dir: Directory containing the repacked DataFrames for THINGS and NSD images
                           (with CLIP embeddings and raw images).
                           NOTE: extra 3 embeddings should be under the same directory with <image_df name>_emb.pkl
            mode: One of 'train', 'val', or 'test' to specify which split to load.
            normalize_fmri: Whether to L2-normalize the fMRI vectors. Default is True.
            image_size: Size of raw images, deprecated used as assertion and dummy only. Default is 224.

            load_images: Whether to load CLIP embeddings (786 dimension one) and raw images. Default is False.

            emb_source: Source for generation targets — 'nsd' or 'things'. Default is 'things'.
            _shared_data: Pre-built lookup dicts shared across splits. 
                           NOTE:(This contains all needed image data and embeddings.)
        """

        # We can only provide embeddings from NSD or THINGS
        # NOTE: the is the image source used to generate bigG/vae/cnx target.
        assert emb_source in ("nsd", "things"), \
            f"emb_source must be 'nsd' or 'things', got '{emb_source}'"
        self.emb_source = emb_source

        # Store path for lazy opening in worker processes (LMDB is NOT fork-safe)
        self.data_dir = data_dir
        self.db = None  # opened lazily in __getitem__

        # Some internal configs
        self.normalize_fmri = normalize_fmri
        self.load_images = load_images
        self.image_size = image_size

        # Temporarily open LMDB to load keys, then close immediately
        _db = lmdb.open(
            data_dir,
            readonly = True,
            lock = False,
            readahead = False,
            meminit = False,
        )
        with _db.begin(write = False) as txn:
            self.keys = [
                key.decode()
                for key, _ in txn.cursor()
                if key.decode().startswith(mode + "_")
            ]
        _db.close()

        # Use pre-built shared data if provided (avoids 3× pkl loading)
        # or, we load it now
        # NOTE: the embeddings' dataframe is shared by train/val/test datasets.
        if _shared_data is None:
            # Warn user in this case
            print(
                f"\033[93mWarning: No shared data provided to {mode} dataset. "
                f"Loading image/embedding data from {images_df_dir} for this split. "
                f"This may cause increased memory usage if multiple splits are loaded.\033[0m"
            )
            # Load data
            _shared_data = EEG_fMRI_Generation_E2E_Dataset.load_shared_data(images_df_dir, emb_source)
        # Set references
        # Image indexes (containing image data)
        self.things_img_idx_to_row = _shared_data["things_img_idx_to_row"]
        self.nsd_img_idx_to_row = _shared_data["nsd_img_idx_to_row"]
        # selected embedding indexes (containing embeddings)
        self.emb_idx_to_row = _shared_data["emb_idx_to_row"]
        # flag for whether image data is available
        # NOTE: This means all data (including embeddings)
        self.has_images = _shared_data["has_images"]
        
        # Warn user if image data not available (dummy would be used)
        if not self.has_images:
            print(
                f"\033[93mWarning: Image data not available for {mode} dataset. "
                f"ALL embeddings and raw images will be returned as zero vectors.\033[0m"
            )

    def __len__(self) -> int:
        # Just the length of the keys for the current mode
        return len(self.keys)

    def __getitem__(self, idx: int) \
        -> typing.Tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray]:

        # Get the key for the current index
        key = self.keys[idx]

        # Lazy-open LMDB in each worker process (LMDB mmap is NOT fork-safe)
        if self.db is None:
            self.db = lmdb.open(
                self.data_dir,
                readonly = True,
                lock = False,
                readahead = False,
                meminit = False,
            )

        # Load the data from LMDB using the key
        with self.db.begin(write = False) as txn:
            data: dict = pickle.loads(txn.get(key.encode()))

        # Extract EEG, fMRI, label, and image indices
        EEG = data["eeg"]  # (1, 63, 200 / 250)
        fMRI = data["fmri"]  # (1, 4096)
        label = data["label"]
        things_img_idx = data["things_img_idx"]
        nsd_img_idx = data["nsd_img_idx"]

        # some assertions to make sure the data shapes are correct
        # REMOVE if causing issues, but good for debugging
        assert EEG.shape == (1, 63, 200) or EEG.shape == (1, 63, 250),\
            f"Unexpected EEG shape: {EEG.shape} for key {key}"
        assert fMRI.shape == (1, 4096), \
            f"Unexpected fMRI shape: {fMRI.shape} for key {key}"

        # Normalize EEG
        # TODO: review this
        EEG = EEG / 100

        # Optionally normalize fMRI
        if self.normalize_fmri:
            fMRI_norm = np.linalg.norm(fMRI)
            if fMRI_norm > 0:
                fMRI = fMRI / fMRI_norm

        # Get CLIP embedding from images DataFrame
        # NOTE: we return zero vectors if image/CLIP data is not available
        #       or we don't need to load them
        things_clip_embed = np.zeros(768, dtype = np.float32)
        nsd_clip_embed = np.zeros(768, dtype = np.float32)
        things_image_data = np.zeros((3, self.image_size, self.image_size), dtype = np.float32)
        nsd_image_data = np.zeros((3, self.image_size, self.image_size), dtype = np.float32)

        clip_target_bigG = np.zeros((256, 1664), dtype = np.float32)
        vae_latents = np.zeros((4, 28, 28), dtype = np.float32)
        cnx_features = np.zeros((49, 512), dtype = np.float32)

        if self.has_images:
            
            # If we wish to load images (image + 768 embedding)
            if self.load_images:

                # Get the row for the current image index
                things_row = self.things_img_idx_to_row.get(things_img_idx, None)
                nsd_row = self.nsd_img_idx_to_row.get(nsd_img_idx, None)

                # Extract CLIP embedding and image data if available
                if "image_embedding" in things_row and things_row["image_embedding"] is not None:
                    things_clip_embed = things_row["image_embedding"].astype(np.float32)
                if "image_embedding" in nsd_row and nsd_row["image_embedding"] is not None:
                    nsd_clip_embed = nsd_row["image_embedding"].astype(np.float32)

                # Extract and process raw image data if available
                if "image_data" in things_row and things_row["image_data"] is not None:
                    img = things_row["image_data"]  # (H, W, 3) uint8
                    # Check datatype and shape before processing
                    assert img.dtype == np.uint8, f"Unexpected image dtype: {img.dtype} for THINGS image index {things_img_idx}"
                    assert img.shape == (self.image_size, self.image_size, 3), f"Unexpected image shape: {img.shape} for THINGS image index {things_img_idx}"
                    # Normalize to [0, 1] and transpose to (3, H, W)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # (3, H, W)
                    things_image_data = img
                if "image_data" in nsd_row and nsd_row["image_data"] is not None:
                    img = nsd_row["image_data"]  # (H, W, 3) uint8
                    # Check datatype and shape before processing
                    assert img.dtype == np.uint8, f"Unexpected image dtype: {img.dtype} for NSD image index {nsd_img_idx}"
                    assert img.shape == (self.image_size, self.image_size, 3), f"Unexpected image shape: {img.shape} for NSD image index {nsd_img_idx}"
                    # Normalize to [0, 1] and transpose to (3, H, W)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # (3, H, W)
                    nsd_image_data = img

            # Load pre-computed generation targets based on emb_source setting
            # Pre-computed generation targets (zero fallback if unavailable)
            # if image data available
            # select corresponding embedding index
            if self.emb_source == "nsd":
                emb_idx = nsd_img_idx
            else:  # "things"
                emb_idx = things_img_idx
            # load from lookup dict built from embedding dataframe
            row = self.emb_idx_to_row.get(emb_idx, None)
            # assert row not None and has needed fields
            assert row is not None, f"Missing embedding row for index {emb_idx}"
            assert "clip_bigG_embeddings" in row and "vae_latents" in row and "cnx_features" in row, \
                f"Missing required fields in embedding row for index {emb_idx}"
            # Load embeddings
            if "clip_bigG_embeddings" in row and row["clip_bigG_embeddings"] is not None:
                clip_target_bigG = row["clip_bigG_embeddings"].astype(np.float32)
            if "vae_latents" in row and row["vae_latents"] is not None:
                vae_latents = row["vae_latents"].astype(np.float32)
            if "cnx_features" in row and row["cnx_features"] is not None:
                cnx_features = row["cnx_features"].astype(np.float32)

        # return everything
        return EEG, fMRI, label, \
               things_img_idx, nsd_img_idx, \
               things_clip_embed, nsd_clip_embed, \
               things_image_data, nsd_image_data, \
               clip_target_bigG, vae_latents, cnx_features

    def collate(self, batch: typing.List) \
        -> typing.Tuple[torch.Tensor, ...]:

        EEG = np.array([x[0] for x in batch]).squeeze()
        fMRI = np.array([x[1] for x in batch]).squeeze()
        label = np.array([x[2] for x in batch])
        things_img_idx = np.array([x[3] for x in batch])
        nsd_img_idx = np.array([x[4] for x in batch])
        things_clip_embed = np.array([x[5] for x in batch])
        nsd_clip_embed = np.array([x[6] for x in batch])
        things_image_data = np.array([x[7] for x in batch])
        nsd_image_data = np.array([x[8] for x in batch])
        clip_target_bigG = np.array([x[9] for x in batch])
        vae_latents = np.array([x[10] for x in batch])
        cnx_features = np.array([x[11] for x in batch])

        # Reshape EEG to (B, C, 1, T)
        EEG = EEG.reshape(EEG.shape[0], EEG.shape[1], 1, EEG.shape[2])

        return (
            torch.from_numpy(EEG).float(),
            torch.from_numpy(fMRI).float(),
            torch.from_numpy(label).long(),
            torch.from_numpy(things_img_idx).long(),
            torch.from_numpy(nsd_img_idx).long(),
            torch.from_numpy(things_clip_embed).float(),
            torch.from_numpy(nsd_clip_embed).float(),
            torch.from_numpy(things_image_data).float(),
            torch.from_numpy(nsd_image_data).float(),
            torch.from_numpy(clip_target_bigG).float(),
            torch.from_numpy(vae_latents).float(),
            torch.from_numpy(cnx_features).float(),
        )

    @staticmethod
    def load_shared_data(images_df_dir: str, emb_source: str) -> dict:

        """
        Load image DataFrames and embeddings once, 
        returning lookup dicts to share across splits.
        """

        # Get paths to dataframe
        things_images_df_path = os.path.join(images_df_dir, "./things_images_df.pkl")
        nsd_images_df_path = os.path.join(images_df_dir, "./nsd_images_df.pkl")
        # Make embedding pkl path
        # join <source_images_df_path>_emb.pkl based on emb_source
        emb_df_path = (things_images_df_path if emb_source == "things" else nsd_images_df_path).replace("_images_df.pkl", "_images_emb.pkl")

        # Initialize empty lookup dicts and flag
        # NOTE: we only load the requested embeddings
        things_img_idx_to_row = {}
        nsd_img_idx_to_row = {}
        emb_idx_to_row = {}
        has_images = False

        # Load image DataFrames and build lookup dicts if both files exist
        if os.path.exists(things_images_df_path) and \
            os.path.exists(nsd_images_df_path) and \
            os.path.exists(emb_df_path):

            # Load the DataFrames
            # NOTE: this is the repacked image dataframe
            things_images_df = pd.read_pickle(things_images_df_path)
            nsd_images_df = pd.read_pickle(nsd_images_df_path)

            # Build index lookup for fast access
            # NOTE: {image_index : row}
            #       row is the single row in dataframe
            for _, row in things_images_df.iterrows():
                things_img_idx_to_row[int(row["image_index"])] = row
            for _, row in nsd_images_df.iterrows():
                nsd_img_idx_to_row[int(row["image_index"])] = row

            # verbose
            print(
                f"\033[92mSuccessfully loaded THINGS and NSD images DataFrames from {images_df_dir}.\033[0m"
            )

            # Load 3 extra embedding pickle
            emb_df = pd.read_pickle(emb_df_path)
            for _ , row in emb_df.iterrows():
                emb_idx_to_row[int(row["image_index"])] = row
            print(
                f"\033[92mLoaded {emb_source} embeddings DataFrame from {emb_df_path} ({len(emb_df)} entries).\033[0m"
            )

            # set flag
            has_images = True

        else:
            if not os.path.exists(things_images_df_path):
                print(
                    f"\033[91mWarning: THINGS images DataFrame not found at {things_images_df_path}.\033[0m"
                )
            if not os.path.exists(nsd_images_df_path):
                print(
                    f"\033[91mWarning: NSD images DataFrame not found at {nsd_images_df_path}.\033[0m"
                )
            if not os.path.exists(emb_df_path):
                print(
                    f"\033[91mWarning: Embeddings DataFrame for {emb_source} not found at {emb_df_path}.\033[0m"
                )
            # Print not available warning
            print(
                f"\033[93mImage data and/or embeddings not available. "
                f"All embeddings and raw images will be returned as zero vectors.\033[0m"
            )

        return {
            "things_img_idx_to_row": things_img_idx_to_row,
            "nsd_img_idx_to_row": nsd_img_idx_to_row,
            "emb_idx_to_row": emb_idx_to_row,
            "has_images": has_images,
        }

def get_generation_data_loader(
    datasets_dir: str,
    images_df_dir: str = default_images_df_path,
    batch_size: int = 32,
    normalize_fmri: bool = True,
    load_images: bool = False,
    num_workers: int = 0,
    emb_source: str = "things",
) -> typing.Dict[str, DataLoader]:
    
    """
    Create data loaders for the generation E2E pipeline.
    Args:
        datasets_dir: Path to the LMDB dataset directory containing aligned EEG-fMRI data.
        images_df_path: Directory containing the repacked DataFrames for THINGS and NSD images (with CLIP embeddings and raw images).
        batch_size: Batch size for the data loaders. Default is 32.
        normalize_fmri: Whether to L2-normalize the fMRI vectors. Default is True.
        load_images: Whether to load CLIP embeddings and raw images (for blurry reconstruction target). Default is False.
        num_workers: Number of worker processes for data loading. Default is 0 (no multiprocessing).
        emb_source: Source for generation targets — 'nsd' or 'things'. Default is 'things'.
    """
    # Load pkl data once and share across all splits
    print("Loading shared image/embedding data...")
    shared_data = EEG_fMRI_Generation_E2E_Dataset.load_shared_data(images_df_dir, emb_source)

    # Set up datasets
    print(f"Setting up training dataset:")
    train_set = EEG_fMRI_Generation_E2E_Dataset(
        datasets_dir,
        images_df_dir,
        mode = "train",
        normalize_fmri = normalize_fmri,
        load_images = load_images,
        emb_source = emb_source,
        _shared_data = shared_data,
    )
    print(f"Setting up validation dataset:")
    val_set = EEG_fMRI_Generation_E2E_Dataset(
        datasets_dir,
        images_df_dir,
        mode = "val",
        normalize_fmri = normalize_fmri,
        load_images = load_images,
        emb_source = emb_source,
        _shared_data = shared_data,
    )
    print(f"Setting up testing dataset:")
    test_set = EEG_fMRI_Generation_E2E_Dataset(
        datasets_dir,
        images_df_dir,
        mode = "test",
        normalize_fmri = normalize_fmri,
        load_images = load_images,
        emb_source = emb_source,
        _shared_data = shared_data,
    )

    print(
        f"Dataset sizes: train = {len(train_set)}, "
        f"val = {len(val_set)}, test = {len(test_set)}"
    )
    print(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}")
    print(f"fMRI normalization: {'enabled' if normalize_fmri else 'disabled'}")

    data_loader = {
        "train": DataLoader(
            train_set,
            batch_size = batch_size,
            collate_fn = train_set.collate,
            shuffle = True,
            num_workers = num_workers,
        ),
        "val": DataLoader(
            val_set,
            batch_size = batch_size,
            collate_fn = val_set.collate,
            shuffle = False,
            num_workers = num_workers,
        ),
        "test": DataLoader(
            test_set,
            batch_size = batch_size,
            collate_fn = test_set.collate,
            shuffle = False,
            num_workers = num_workers,
        ),
    }
    return data_loader


# Test code
if __name__ == "__main__":

    datasets_dir = \
        "datasets/processed/eeg_fmri_align_datasets/things_sub-01_nsd_sub-01"
    images_df_dir = "datasets/processed"
    batch_size = 16
    
    data_loader = get_generation_data_loader(
        datasets_dir, images_df_dir, batch_size, emb_source = 'things'
    )

    for (
        EEG_batch,
        fMRI_batch,
        label_batch,
        things_idx,
        nsd_idx,
        things_clip_batch,
        nsd_clip_batch,
        things_image_batch,
        nsd_image_batch,
        clip_target_bigG_batch,
        vae_latents_batch,
        cnx_features_batch
    ) in data_loader["train"]:
        print(f"EEG batch shape: {EEG_batch.shape}")
        print(f"fMRI batch shape: {fMRI_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        print(f"THINGS image indices (first 5): {things_idx[ : 5]}")
        print(f"NSD image indices (first 5): {nsd_idx[ : 5]}")
        print(f"THINGS CLIP batch shape: {things_clip_batch.shape}")
        print(f"NSD CLIP batch shape: {nsd_clip_batch.shape}")
        print(f"THINGS image batch shape: {things_image_batch.shape}")
        print(f"NSD image batch shape: {nsd_image_batch.shape}")
        print(f"CLIP target bigG batch shape: {clip_target_bigG_batch.shape}")
        print(f"VAE latents batch shape: {vae_latents_batch.shape}")
        print(f"ConvNeXt features batch shape: {cnx_features_batch.shape}")
        break