"""
Standalone evaluation of image reconstruction quality.
Computes 8 metrics following MindEyeV2's final_evaluations.ipynb methodology.

Usage (run from project root):
    python -m eval.evaluate_reconstruction \\
        --images_dir runs/.../images \\
        [--output_dir runs/.../] \\
        [--device cuda:0] \\
        [--batch_size 40] \\
        [--use_blurry_enhancement false]
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import scipy as sp


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image reconstruction quality (8 MindEyeV2 metrics)")
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing {split}_sample_NNNNN_{orig,recon,blurry}.png')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save eval_metrics.json. Defaults to parent of images_dir.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='Batch size for model forward passes')
    parser.add_argument('--use_blurry_enhancement',
                        type=lambda x: x.lower() == 'true', default=False,
                        help='Blend recon*0.75 + blurry*0.25 (MindEyeV2 enhanced mode)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def discover_samples(images_dir):
    """
    Scan images_dir for files matching {split}_sample_{index}_{orig|recon|blurry}.png.
    Returns a list of dicts sorted by sample index, each with keys:
        'index', 'orig', 'recon', 'blurry' (blurry may be None).
    """
    pattern = re.compile(r'^(.+)_sample_(\d+)_(orig|recon|blurry)\.png$')
    groups = {}
    for fname in os.listdir(images_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(2))
            img_type = m.group(3)
            if idx not in groups:
                groups[idx] = {}
            groups[idx][img_type] = os.path.join(images_dir, fname)

    samples = []
    for idx in sorted(groups.keys()):
        g = groups[idx]
        if 'orig' not in g or 'recon' not in g:
            raise ValueError(f"Sample {idx} is missing 'orig' or 'recon' file.")
        samples.append({
            'index': idx,
            'orig': g['orig'],
            'recon': g['recon'],
            'blurry': g.get('blurry', None),
        })
    return samples


def load_images_as_tensors(samples, image_types, target_size=256):
    """
    Load PNG images for each image_type, resize to target_size x target_size,
    and return dict mapping type -> tensor of shape (N, 3, target_size, target_size)
    in float32 [0, 1] range.
    """
    to_tensor = transforms.Compose([
        transforms.Resize(
            (target_size, target_size),
            interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    result = {t: [] for t in image_types}
    for sample in tqdm(samples, desc="Loading images"):
        for t in image_types:
            path = sample[t]
            if path is None:
                raise ValueError(f"Sample {sample['index']} missing '{t}' image.")
            img = Image.open(path).convert('RGB')
            result[t].append(to_tensor(img))

    return {t: torch.stack(result[t], dim=0) for t in image_types}


# ---------------------------------------------------------------------------
# 2-way identification helper (exact MindEyeV2 implementation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess,
                           device, feature_layer=None, batch_size=40):
    """
    MindEyeV2 2-way identification metric.
    Extracts features in batches, then computes the full NxN corrcoef.
    Returns a scalar in [0, 1] (higher = better).
    """
    def extract(images):
        feats = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(
                [preprocess(img) for img in images[i:i + batch_size]], dim=0
            ).to(device)
            out = model(batch)
            if feature_layer is not None:
                out = out[feature_layer]
            feats.append(out.float().flatten(1).cpu())
        return torch.cat(feats, dim=0).numpy()

    preds = extract(all_recons)
    reals = extract(all_images)

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(all_images) - 1)
    return float(perf)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_pixcorr(all_images, all_recons):
    """PixCorr: resize to 425, flatten, average np.corrcoef diagonal."""
    preprocess = transforms.Resize(
        425, interpolation=transforms.InterpolationMode.BILINEAR)

    imgs_flat = preprocess(all_images).reshape(len(all_images), -1).cpu()
    recs_flat = preprocess(all_recons).reshape(len(all_recons), -1).cpu()

    corrsum = 0.0
    for i in tqdm(range(len(all_images)), desc="PixCorr"):
        corrsum += np.corrcoef(imgs_flat[i].numpy(), recs_flat[i].numpy())[0][1]
    return float(corrsum / len(all_images))


def compute_ssim(all_images, all_recons):
    """SSIM: resize to 425, grayscale, skimage ssim with gaussian weights."""
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as sk_ssim
    import inspect

    preprocess = transforms.Resize(
        425, interpolation=transforms.InterpolationMode.BILINEAR)

    img_gray = rgb2gray(preprocess(all_images).permute(0, 2, 3, 1).cpu().numpy())
    rec_gray = rgb2gray(preprocess(all_recons).permute(0, 2, 3, 1).cpu().numpy())

    # Handle scikit-image API: 'multichannel' was removed in >=0.19
    ssim_kwargs = dict(gaussian_weights=True, sigma=1.5,
                       use_sample_covariance=False, data_range=1.0)
    sig = inspect.signature(sk_ssim)
    if 'channel_axis' in sig.parameters:
        pass  # grayscale, no channel_axis needed
    # else: old API, multichannel not needed for grayscale either

    scores = []
    for im, rec in tqdm(zip(img_gray, rec_gray), total=len(all_images), desc="SSIM"):
        scores.append(sk_ssim(rec, im, **ssim_kwargs))
    return float(np.mean(scores))


def compute_alexnet_metrics(all_images, all_recons, device, batch_size):
    """AlexNet(2) and AlexNet(5) 2-way identification."""
    from torchvision.models import alexnet, AlexNet_Weights

    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights),
        return_nodes=['features.4', 'features.11']
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("  AlexNet(2)...")
    a2 = two_way_identification(all_recons, all_images, alex_model,
                                preprocess, device, 'features.4', batch_size)
    print("  AlexNet(5)...")
    a5 = two_way_identification(all_recons, all_images, alex_model,
                                preprocess, device, 'features.11', batch_size)

    del alex_model
    torch.cuda.empty_cache()
    return a2, a5


def compute_inception_metric(all_images, all_recons, device, batch_size):
    """InceptionV3 2-way identification (avgpool layer)."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    score = two_way_identification(all_recons, all_images, inception_model,
                                   preprocess, device, 'avgpool', batch_size)
    del inception_model
    torch.cuda.empty_cache()
    return score


def compute_clip_metric(all_images, all_recons, device, batch_size):
    """CLIP ViT-L/14 image features 2-way identification."""
    import clip

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    score = two_way_identification(all_recons, all_images,
                                   clip_model.encode_image,
                                   preprocess, device, None, batch_size)
    del clip_model
    torch.cuda.empty_cache()
    return score


@torch.no_grad()
def extract_features_batched(images, model, preprocess, device,
                              feature_layer, batch_size):
    """Extract model features in batches, return (N, D) numpy array."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(
            [preprocess(img) for img in images[i:i + batch_size]], dim=0
        ).to(device)
        out = model(batch)[feature_layer]
        feats.append(out.float().flatten(1).cpu())
    return torch.cat(feats, dim=0).numpy()


def compute_effnet_metric(all_images, all_recons, device, batch_size):
    """EfficientNet-B1 avgpool pairwise scipy correlation distance."""
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    gt = extract_features_batched(all_images, eff_model, preprocess,
                                   device, 'avgpool', batch_size)
    fake = extract_features_batched(all_recons, eff_model, preprocess,
                                    device, 'avgpool', batch_size)

    dist = np.array([sp.spatial.distance.correlation(gt[i], fake[i])
                     for i in range(len(gt))]).mean()
    del eff_model
    torch.cuda.empty_cache()
    return float(dist)


def compute_swav_metric(all_images, all_recons, device, batch_size):
    """SwAV ResNet50 avgpool pairwise scipy correlation distance."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(
        swav_model, return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    gt = extract_features_batched(all_images, swav_model, preprocess,
                                   device, 'avgpool', batch_size)
    fake = extract_features_batched(all_recons, swav_model, preprocess,
                                    device, 'avgpool', batch_size)

    dist = np.array([sp.spatial.distance.correlation(gt[i], fake[i])
                     for i in range(len(gt))]).mean()
    del swav_model
    torch.cuda.empty_cache()
    return float(dist)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if torch.cuda.is_available() and args.device.startswith('cuda'):
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available, running on CPU (will be slow).")

    # ---- Discover and load images ------------------------------------------
    print(f"\nScanning: {args.images_dir}")
    samples = discover_samples(args.images_dir)
    print(f"Found {len(samples)} samples.")

    image_types = ['orig', 'recon']
    has_blurry = all(s['blurry'] is not None for s in samples)
    if args.use_blurry_enhancement:
        if not has_blurry:
            print("Warning: --use_blurry_enhancement requested but no blurry images found. Skipping blend.")
            args.use_blurry_enhancement = False
        else:
            image_types.append('blurry')

    tensors = load_images_as_tensors(samples, image_types, target_size=256)
    all_images = tensors['orig']    # (N, 3, 256, 256)
    all_recons = tensors['recon']   # (N, 3, 256, 256)

    if args.use_blurry_enhancement:
        print("Applying enhanced blend: recon*0.75 + blurry*0.25")
        all_recons = all_recons * 0.75 + tensors['blurry'] * 0.25

    print(f"\nall_images: {all_images.shape}, all_recons: {all_recons.shape}")

    # ---- Compute metrics ---------------------------------------------------
    results = {}

    print("\n--- PixCorr ---")
    results['PixCorr'] = compute_pixcorr(all_images, all_recons)
    print(f"  {results['PixCorr']:.6f}")

    print("\n--- SSIM ---")
    results['SSIM'] = compute_ssim(all_images, all_recons)
    print(f"  {results['SSIM']:.6f}")

    print("\n--- AlexNet(2) & AlexNet(5) ---")
    a2, a5 = compute_alexnet_metrics(all_images, all_recons, device, args.batch_size)
    results['AlexNet(2)'] = a2
    results['AlexNet(5)'] = a5
    print(f"  AlexNet(2): {a2:.6f}")
    print(f"  AlexNet(5): {a5:.6f}")

    print("\n--- InceptionV3 ---")
    results['InceptionV3'] = compute_inception_metric(
        all_images, all_recons, device, args.batch_size)
    print(f"  {results['InceptionV3']:.6f}")

    print("\n--- CLIP (ViT-L/14, image 2-way) ---")
    results['CLIP'] = compute_clip_metric(
        all_images, all_recons, device, args.batch_size)
    print(f"  {results['CLIP']:.6f}")

    print("\n--- EffNet-B ---")
    results['EffNet-B'] = compute_effnet_metric(
        all_images, all_recons, device, args.batch_size)
    print(f"  {results['EffNet-B']:.6f}  (distance, lower = better)")

    print("\n--- SwAV ---")
    results['SwAV'] = compute_swav_metric(
        all_images, all_recons, device, args.batch_size)
    print(f"  {results['SwAV']:.6f}  (distance, lower = better)")

    # ---- Summary table -----------------------------------------------------
    print("\n" + "=" * 45)
    print(f"{'Metric':<15} {'Value':>12}  {'Direction':>10}")
    print("-" * 45)
    directions = {
        'PixCorr': 'higher↑', 'SSIM': 'higher↑',
        'AlexNet(2)': 'higher↑', 'AlexNet(5)': 'higher↑',
        'InceptionV3': 'higher↑', 'CLIP': 'higher↑',
        'EffNet-B': 'lower↓', 'SwAV': 'lower↓',
    }
    for k, v in results.items():
        print(f"{k:<15} {v:>12.6f}  {directions[k]:>10}")
    print("=" * 45)

    # ---- Save JSON ---------------------------------------------------------
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.images_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "eval_metrics.json")

    save_data = {
        'num_samples': len(samples),
        'images_dir': os.path.abspath(args.images_dir),
        'use_blurry_enhancement': args.use_blurry_enhancement,
        'metrics': {k: float(v) for k, v in results.items()},
        'notes': {
            'PixCorr': 'higher is better',
            'SSIM': 'higher is better',
            'AlexNet(2)': '2-way identification, higher is better',
            'AlexNet(5)': '2-way identification, higher is better',
            'InceptionV3': '2-way identification, higher is better',
            'CLIP': '2-way identification (image features), higher is better',
            'EffNet-B': 'correlation distance, lower is better',
            'SwAV': 'correlation distance, lower is better',
        },
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved -> {output_path}")


if __name__ == '__main__':
    main()
