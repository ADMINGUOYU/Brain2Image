# EEG-fMRI End-to-End inference script.
# Loads a trained EEG_fMRI_E2E checkpoint, runs a dataset split through the
# full generation pipeline, and saves reconstructed images + alignment metrics.
#
# Usage (package-style, run from project root):
#   python -m infer.infer_EEG_fMRI_e2e \
#       --model_dir <ckpt.pth> \
#       --datasets_dir <lmdb_dir> \
#       --unclip_ckpt <unclip6_epoch0_step110000.ckpt> \
#       --unclip_config <MindEyeV2/src/generative_models/configs/unclip6.yaml> \
#       [--autoenc_ckpt <sd_image_var_autoenc.pth>] \
#       [--output_dir ./inference_out] [--max_batches 2]
#
# Output layout:
#   <output_dir>/
#     images/
#       <split>_sample_00000_orig.png      (original image, when available)
#       <split>_sample_00000_recon.png     (768x768 unCLIP reconstruction)
#       <split>_sample_00000_blurry.png    (256x256 VAE blurry recon, if enabled)
#       ...
#     metrics.json                         (MSE, CosSim, Top1, Top10 retrieval)

import os
import sys
import json
import argparse
import typing
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Add MindEyeV2/src to sys.path so that `generative_models` package is importable,
# and MindEyeV2/src/generative_models/ so that internal `import sgm` calls within
# the sgm package itself resolve correctly (mirrors the notebook's sys.path.append).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINDEYE_SRC = os.path.join(_PROJECT_ROOT, "MindEyeV2", "src")
_GENERATIVE_MODELS = os.path.join(_MINDEYE_SRC, "generative_models")
for _p in (_MINDEYE_SRC, _GENERATIVE_MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from diffusers import AutoencoderKL

from model.EEG_fMRI_e2e import EEG_fMRI_E2E
from data.EEG_fMRI_generation_e2e_dataset import get_generation_data_loader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG-fMRI E2E inference")

    # ── Inference-specific ────────────────────────────────────────────────
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained E2E checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='./inference_out',
                        help='Directory where images and metrics are saved')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of unCLIP reconstructions per EEG sample')
    parser.add_argument('--prior_timesteps', type=int, default=20,
                        help='Diffusion prior denoising steps (1–100)')
    parser.add_argument('--unclip_ckpt', type=str, default=None,
                        help='Path to unclip6_epoch0_step110000.ckpt')
    parser.add_argument('--unclip_config', type=str, default=None,
                        help='Path to generative_models/configs/unclip6.yaml')
    parser.add_argument('--autoenc_ckpt', type=str, default=None,
                        help='Path to sd_image_var_autoenc.pth (blurry recon)')
    parser.add_argument('--save_blurry', type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Whether to also decode and save blurry reconstructions')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Stop after this many batches (quick test mode)')

    # ── General ───────────────────────────────────────────────────────────
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # ── Encoder backbone (must match the training run) ────────────────────
    parser.add_argument('--backbone', type=str, default='ATMS',
                        choices=['CBraMod', 'ATMS'])
    parser.add_argument('--use_pretrained_weights',
                        type=lambda x: x.lower() == 'true', default=False,
                        help='Ignored at inference; kept for symmetry with training args')
    parser.add_argument('--foundation_dir', type=str, default=None)

    # ATMS parameters
    parser.add_argument('--atms_emb_size', type=int, default=40)
    parser.add_argument('--out_mlp_dim', type=int, default=4096)
    parser.add_argument('--atms_drop_proj', type=float, default=0.5)
    parser.add_argument('--atms_d_model', type=int, default=250)
    parser.add_argument('--atms_n_heads', type=int, default=4)
    parser.add_argument('--atms_d_ff', type=int, default=256)
    parser.add_argument('--atms_dropout', type=float, default=0.25)
    parser.add_argument('--atms_factor', type=int, default=1)

    # CBraMod parameters
    parser.add_argument('--pooling_type', type=str, default='flatten',
                        choices=['flatten', 'attention', 'multitoken_vit'])
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--num_tokens', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=4)

    # ── Dataset / embedding ───────────────────────────────────────────────
    parser.add_argument('--datasets_dir', type=str, required=True,
                        help='Path to LMDB dataset directory')
    parser.add_argument('--images_df_dir', type=str, default='datasets/processed')
    parser.add_argument('--normalize_fmri',
                        type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--emb_source', type=str, default='things',
                        choices=['nsd', 'things'])

    # ── Alignment / generation loss params (for model_config construction) ─
    parser.add_argument('--mse_scale', type=float, default=1.0)
    parser.add_argument('--infonce_scale', type=float, default=0.2)
    parser.add_argument('--proto_distill_scale', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--blurry_recon',
                        type=lambda x: x.lower() == 'true', default=True,
                        help='Must match the training setting. '
                             'Auto-detected from checkpoint if omitted.')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def peek_checkpoint_params(ckpt_path: str) -> typing.Dict:
    """Load only the 'parameters' dict from a checkpoint without loading weights."""
    loaded = torch.load(ckpt_path, map_location='cpu')
    return loaded.get('parameters', {})


def load_unclip_engine(config_path: str, ckpt_path: str,
                        device: torch.device) -> typing.Tuple:
    """
    Instantiate the unCLIP DiffusionEngine, load weights, and compute
    the vector_suffix conditioning vector from a dummy 768×768 batch.

    Note: sgm hardcodes attn_mode="softmax-xformers" for every transformer
    block. When xformers is absent it logs one warning per block. We suppress
    those here (ERROR level); real errors still surface.

    Returns:
        diffusion_engine: loaded and eval'd DiffusionEngine
        vector_suffix:    (1, 1024) size-conditioning tensor on `device`
    """
    import logging
    logging.getLogger("sgm.modules.attention").setLevel(logging.ERROR)

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]

    first_stage_config = unclip_params["first_stage_config"]
    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'

    diffusion_engine = DiffusionEngine(
        network_config=unclip_params["network_config"],
        denoiser_config=unclip_params["denoiser_config"],
        first_stage_config=first_stage_config,
        conditioner_config=unclip_params["conditioner_config"],
        sampler_config=unclip_params["sampler_config"],
        scale_factor=unclip_params["scale_factor"],
        disable_first_stage_autocast=unclip_params["disable_first_stage_autocast"],
    )
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    print(f"Loaded unCLIP engine from {ckpt_path}")

    # Compute the size-conditioning vector once via a dummy pass
    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
    }
    with torch.no_grad():
        out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print(f"vector_suffix shape: {vector_suffix.shape}")

    return diffusion_engine, vector_suffix


def load_autoenc(ckpt_path: str, device: torch.device) -> AutoencoderKL:
    """
    Build the SD image VAE (fixed architecture from MindEyeV2) and load weights.
    Used for decoding blurry_latents back to pixel space.
    """
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D',
                          'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D',
                        'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    autoenc.load_state_dict(ckpt)
    autoenc.eval().requires_grad_(False)
    autoenc.to(device)
    print(f"Loaded autoencoder from {ckpt_path}")
    return autoenc


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

def unclip_recon(x: torch.Tensor,
                 diffusion_engine: DiffusionEngine,
                 vector_suffix: torch.Tensor,
                 num_samples: int,
                 device: torch.device,
                 offset_noise_level: float = 0.04) -> torch.Tensor:
    """
    Reconstruct images from a single set of CLIP patch tokens via unCLIP.

    Args:
        x:               (1, 256, 1664) CLIP tokens for one sample
        diffusion_engine: loaded DiffusionEngine
        vector_suffix:   (1, 1024) size-conditioning vector
        num_samples:     how many reconstructions to produce
        device:          CUDA device
        offset_noise_level: noise level for latent conditioning

    Returns:
        samples: (num_samples, 3, H, W) float tensor in [0, 1]
    """
    assert x.ndim == 3 and x.shape[0] == 1, \
        f"Expected x shape (1, 256, 1664), got {x.shape}"

    x = x.to(device)
    with torch.no_grad(), \
         torch.cuda.amp.autocast(dtype=torch.float16), \
         diffusion_engine.ema_scope():

        z = torch.randn(num_samples, 4, 96, 96, device=device)

        c = {
            "crossattn": x.repeat(num_samples, 1, 1),
            "vector": vector_suffix.repeat(num_samples, 1),
        }
        uc = {
            "crossattn": torch.randn_like(x).repeat(num_samples, 1, 1),
            "vector": vector_suffix.repeat(num_samples, 1),
        }

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(
            diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp(samples_x * 0.8 + 0.2, min=0.0, max=1.0)

    return samples.float()


def save_tensor_as_png(tensor: torch.Tensor, path: str):
    """Save a (3, H, W) float tensor in [0, 1] as a PNG file."""
    pil_img = transforms.ToPILImage()(tensor.clamp(0, 1).cpu().float())
    pil_img.save(path)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: EEG_fMRI_E2E,
    data_loader: torch.utils.data.DataLoader,
    diffusion_engine: typing.Optional[DiffusionEngine],
    vector_suffix: typing.Optional[torch.Tensor],
    autoenc: typing.Optional[AutoencoderKL],
    args: argparse.Namespace,
    device: torch.device,
    images_dir: str,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate over data_loader, generate images, and accumulate embeddings
    for alignment metric computation.

    Returns:
        all_eeg_embeds: (N, 4096) EEG embeddings (cpu)
        all_fmri:       (N, 4096) fMRI targets    (cpu)
    """
    model.eval()
    sample_idx = 0
    all_eeg_embeds = []
    all_fmri = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(data_loader, desc=f"Inferring [{args.split}]")):

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            (EEG, fMRI, label,
             things_img_idx, nsd_img_idx,
             _, _,
             things_image_data, nsd_image_data,
             clip_target, vae_latents, cnx_features, _) = batch

            EEG = EEG.to(device)
            fMRI = fMRI.to(device)

            # Stage 1: EEG encoder → 4096-dim embeddings
            with torch.amp.autocast(device.type):
                eeg_embeds = model.forward_encoder(EEG)          # (B, 4096)
                gen_outputs = model.forward_generation(eeg_embeds)

            backbone = gen_outputs['backbone']                    # (B, 256, 1664)
            blurry_latents = gen_outputs.get('blurry_latents')   # (B, 4, 28, 28) or None

            # Stage 2: diffusion prior denoising
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape,
                text_cond=dict(text_embed=backbone),
                cond_scale=1.0,
                timesteps=args.prior_timesteps,
            )   # (B, 256, 1664)

            # Stage 3: per-sample image generation and saving
            B = EEG.shape[0]
            for i in range(B):
                # Save original image if the dataset provided a non-zero one
                orig_img = things_image_data[i]   # (3, 224, 224)
                if orig_img.abs().sum() > 0:
                    save_tensor_as_png(
                        orig_img,
                        os.path.join(
                            images_dir,
                            f"{args.split}_sample_{sample_idx:05d}_orig.png")
                    )

                # unCLIP reconstruction (only when engine is loaded)
                if diffusion_engine is not None:
                    samples = unclip_recon(
                        prior_out[[i]].float(),
                        diffusion_engine,
                        vector_suffix,
                        num_samples=args.num_samples,
                        device=device,
                    )   # (num_samples, 3, H, W)
                    for s in range(args.num_samples):
                        suffix = f"_s{s}" if args.num_samples > 1 else ""
                        save_tensor_as_png(
                            samples[s],
                            os.path.join(
                                images_dir,
                                f"{args.split}_sample_{sample_idx:05d}{suffix}_recon.png")
                        )

                # Blurry VAE reconstruction
                if (args.save_blurry and autoenc is not None
                        and blurry_latents is not None):
                    latent = blurry_latents[[i]].half()    # (1, 4, 28, 28)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        blurry_decoded = autoenc.decode(latent / 0.18215).sample
                    blurry = (blurry_decoded.float() / 2 + 0.5).clamp(0, 1)
                    save_tensor_as_png(
                        blurry[0],
                        os.path.join(
                            images_dir,
                            f"{args.split}_sample_{sample_idx:05d}_blurry.png")
                    )

                sample_idx += 1

            # Accumulate embeddings (cpu to save GPU memory)
            all_eeg_embeds.append(eeg_embeds.cpu())
            all_fmri.append(fMRI.cpu())

    return (
        torch.cat(all_eeg_embeds, dim=0),
        torch.cat(all_fmri, dim=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ── Auto-detect blurry_recon from checkpoint ──────────────────────────
    print(f"Peeking at checkpoint: {args.model_dir}")
    saved_params = peek_checkpoint_params(args.model_dir)
    blurry_recon = saved_params.get('blurry_recon', args.blurry_recon)
    print(f"blurry_recon (from checkpoint): {blurry_recon}")

    # ── Build model config ────────────────────────────────────────────────
    if args.backbone == 'CBraMod':
        encoder_config = {
            'encoder_type': 'CBraMod',
            'cuda': args.cuda,
            'use_pretrained_weights': False,
            'foundation_dir': args.foundation_dir,
            'pooling_type': args.pooling_type,
            'attention_heads': args.attention_heads,
            'num_tokens': args.num_tokens,
            'num_transformer_layers': args.num_transformer_layers,
            'num_attention_heads': args.num_attention_heads,
            'mlp_layers': args.mlp_layers,
            'embedding_dim': args.embedding_dim,
        }
    else:  # ATMS
        encoder_config = {
            'encoder_type': 'ATMS',
            'cuda': args.cuda,
            'use_pretrained_weights': False,
            'foundation_dir': args.foundation_dir,
            'num_channels': 63,
            'seq_len': 250,
            'emb_size': args.atms_emb_size,
            'proj_dim': 1024,
            'drop_proj': args.atms_drop_proj,
            'd_model': args.atms_d_model,
            'n_heads': args.atms_n_heads,
            'e_layers': 1,
            'd_ff': args.atms_d_ff,
            'dropout': args.atms_dropout,
            'factor': args.atms_factor,
            'out_mlp_dim': args.out_mlp_dim,
        }

    model_config = {
        'EEG_Encoder': encoder_config,
        'Loss': {
            'mse_scale': args.mse_scale,
            'infonce_scale': args.infonce_scale,
            'proto_distill_scale': args.proto_distill_scale,
            'temperature': args.temperature,
            'normalize_fmri': args.normalize_fmri,
        },
        'Generation': {
            'n_blocks': 4,
            'drop': 0.15,
            'clip_size': 1664,
            'blurry_recon': blurry_recon,
            'clip_scale': 1.0,
            'prior_scale': 30.0,
            'clip_loss_scale': 1.0,
            'blur_scale': 0.5,
            'align_scale': 1.0,
        },
    }

    # ── Build and load model ──────────────────────────────────────────────
    print("Building model...")
    model = EEG_fMRI_E2E(model_config)
    model.load_model(args.model_dir, device)
    model.to(device)
    model.eval()
    print(f"Loaded E2E model from {args.model_dir}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    data_loaders = get_generation_data_loader(
        datasets_dir=args.datasets_dir,
        images_df_dir=args.images_df_dir,
        batch_size=args.batch_size,
        normalize_fmri=args.normalize_fmri,
        load_images=True,   # load original images for side-by-side saving
        num_workers=args.num_workers,
        emb_source=args.emb_source,
        splits=[args.split],
    )
    data_loader = data_loaders[args.split]
    print(f"Using {args.split} split: {len(data_loader.dataset)} samples")

    # ── Load unCLIP engine ────────────────────────────────────────────────
    diffusion_engine = None
    vector_suffix = None
    if args.unclip_ckpt and args.unclip_config:
        print("Loading unCLIP diffusion engine...")
        diffusion_engine, vector_suffix = load_unclip_engine(
            args.unclip_config, args.unclip_ckpt, device
        )
    else:
        print("WARNING: --unclip_ckpt / --unclip_config not provided. "
              "Skipping image generation; only alignment metrics will be saved.")

    # ── Load autoencoder (blurry recon) ───────────────────────────────────
    autoenc = None
    if args.save_blurry and blurry_recon and args.autoenc_ckpt:
        print("Loading VAE autoencoder for blurry reconstruction...")
        autoenc = load_autoenc(args.autoenc_ckpt, device)
    elif args.save_blurry and blurry_recon and not args.autoenc_ckpt:
        print("NOTE: --autoenc_ckpt not provided; blurry reconstructions will be skipped.")

    # ── Run inference ─────────────────────────────────────────────────────
    print("Starting inference...")
    all_eeg_embeds, all_fmri = run_inference(
        model, data_loader, diffusion_engine, vector_suffix, autoenc,
        args, device, images_dir,
    )

    # ── Alignment metrics ─────────────────────────────────────────────────
    print("Computing alignment metrics...")
    all_eeg_embeds = all_eeg_embeds.to(device)
    all_fmri = all_fmri.to(device)
    mse, cos_sim, ret_acc_top1, ret_acc_top10 = \
        model.align_model.get_metrics_for_alignment(
            all_eeg_embeds.squeeze(), all_fmri.squeeze()
        )

    metrics = {
        "split": args.split,
        "num_samples_evaluated": int(all_eeg_embeds.shape[0]),
        "mse": float(mse),
        "cos_sim": float(cos_sim),
        "retrieval_acc_top1": float(ret_acc_top1),
        "retrieval_acc_top10": float(ret_acc_top10),
    }

    print("Alignment metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics  → {metrics_path}")
    print(f"Saved images   → {images_dir}/")


if __name__ == "__main__":
    main()
