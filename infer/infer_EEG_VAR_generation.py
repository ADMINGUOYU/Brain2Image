"""
EEG-VAR Generation Inference Script

Generate images from EEG using trained VAR model.
Supports classifier-free guidance and various sampling strategies.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from model.EEG_VAR_generation import EEG_VAR_Generation
from data.EEG_VAR_generation_dataset import EEG_VAR_Generation_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with EEG-VAR Generation Model')

    # Paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Path to stage 1 checkpoint')
    parser.add_argument('--lmdb_dir', type=str, required=True,
                        help='Path to LMDB directory')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to HDF5 image file')
    parser.add_argument('--vae_ckpt', type=str, default='pretrained/vae_ch160v4096z32.pth',
                        help='Path to VQVAE checkpoint')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='Path to VAR checkpoint (default: pretrained/var_d{depth}.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated images')

    # Model config
    parser.add_argument('--eeg_encoder_type', type=str, default='CBraMod',
                        choices=['CBraMod', 'ATMS'],
                        help='EEG encoder type')
    parser.add_argument('--var_depth', type=int, default=16,
                        help='VAR transformer depth')

    # Inference config
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def get_stage1_config(eeg_encoder_type: str) -> dict:
    """Get stage 1 model configuration."""
    if eeg_encoder_type == 'CBraMod':
        return {
            'EEG_Encoder': {
                'type': 'CBraMod_EEG_Encoder',
                'pooling_type': 'attention',
                'output_dim': 1024
            }
        }
    elif eeg_encoder_type == 'ATMS':
        return {
            'EEG_Encoder': {
                'type': 'ATMS_EEG_Encoder',
                'output_dim': 1024
            }
        }
    else:
        raise ValueError(f"Unknown EEG encoder type: {eeg_encoder_type}")


def get_var_config(depth: int) -> dict:
    """Get VAR model configuration."""
    return {
        'depth': depth,
        'embed_dim': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'norm_eps': 1e-6,
        'shared_aln': False,
        'cond_drop_rate': 0.1,
        'attn_l2_norm': False,
        'patch_nums': (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        'flash_if_available': True,
        'fused_if_available': True,
        'vocab_size': 4096,
        'z_channels': 32,
        'ch': 160,
    }


def save_image(tensor, path):
    """Save image tensor to file."""
    # tensor: (3, H, W) in [0, 1]
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set default VAR checkpoint path if not provided
    if args.var_ckpt is None:
        args.var_ckpt = f'pretrained/var_d{args.var_depth}.pth'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # Create dataset
    print("Creating dataset...")
    dataset = EEG_VAR_Generation_Dataset(
        lmdb_dir=args.lmdb_dir,
        h5_path=args.h5_path,
        mode=args.split,
        image_size=256
    )

    # Limit samples if specified
    if args.max_samples is not None:
        dataset.keys = dataset.keys[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Processing {len(dataset)} samples")

    # Create model
    print("Creating model...")
    stage1_config = get_stage1_config(args.eeg_encoder_type)
    var_config = get_var_config(args.var_depth)

    model = EEG_VAR_Generation(
        stage1_config=stage1_config,
        stage1_ckpt_path=args.stage1_ckpt,
        vae_ckpt_path=args.vae_ckpt,
        var_ckpt_path=args.var_ckpt,
        var_config=var_config,
        device=args.device
    )

    # Load trained checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.var.load_state_dict(checkpoint['var_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    model = model.to(args.device)
    model.eval()

    # Generate images
    print("Generating images...")
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (eeg, gt_images, things_img_indices) in enumerate(tqdm(data_loader)):
            eeg = eeg.to(args.device)
            gt_images = gt_images.to(args.device)

            # Generate images
            gen_images = model.generate(
                eeg=eeg,
                cfg_scale=args.cfg_scale,
                top_k=args.top_k,
                top_p=args.top_p,
                g_seed=args.seed + batch_idx
            )

            # Save images
            for i in range(gen_images.shape[0]):
                # Save generated image
                save_image(
                    gen_images[i],
                    os.path.join(args.output_dir, 'images', f'{sample_idx:05d}_gen.png')
                )

                # Save ground truth image (denormalize from [-1, 1] to [0, 1])
                gt_img = (gt_images[i] + 1) / 2
                save_image(
                    gt_img,
                    os.path.join(args.output_dir, 'images', f'{sample_idx:05d}_gt.png')
                )

                sample_idx += 1

    print(f"\nGeneration completed!")
    print(f"Generated {sample_idx} images")
    print(f"Output directory: {args.output_dir}")

    # Save generation config
    gen_config = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'cfg_scale': args.cfg_scale,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'seed': args.seed,
        'num_samples': sample_idx
    }
    with open(os.path.join(args.output_dir, 'generation_config.json'), 'w') as f:
        json.dump(gen_config, f, indent=2)


if __name__ == '__main__':
    main()
