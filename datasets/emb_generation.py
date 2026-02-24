# Standalone embedding generation script.
# Reads an already-packed images DataFrame pkl (e.g. nsd_images_df.pkl or things_images_df.pkl)
# and produces a separate *_emb.pkl with columns:
#   [image_index, clip_bigG_embeddings, vae_latents, cnx_features]
#
# Usage:
#   python -m datasets.emb_generation --input datasets/processed/nsd_images_df.pkl
#   python -m datasets.emb_generation --input datasets/processed/things_images_df.pkl

"""
Please use 'python -m datasets.emb_generation' to run this script.
"""

import os
os.environ['TRANSFORMERS_CACHE'] = 'datasets/transformers_cache'
os.environ['HF_HOME'] = 'datasets/transformers_cache'
os.environ['HF_HUB_CACHE'] = 'datasets/transformers_cache'
os.environ['TORCH_HOME'] = 'datasets/transformers_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import pandas as pd
import open_clip
from diffusers import AutoencoderKL
from model.autoencoder.convnext import ConvnextXL
from PIL import Image
from tqdm import tqdm

# --------------- Start of configuration --------------- #

CUDA = 0
vae_encode_ckpt_path = 'datasets/NSD/sd_image_var_autoenc_mindeye2.pth'
convnext_xl_ckpt_path = 'datasets/NSD/convnext_xlarge_alpha0.75_fullckpt.pth'

# ---------------- End of configuration ---------------- #


def download_checkpoints():
    """Download model checkpoints if they don't exist."""
    if not os.path.exists(vae_encode_ckpt_path):
        print(f"Downloading Stable Diffusion VAE checkpoint...")
        os.system(f"wget -O {vae_encode_ckpt_path} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/sd_image_var_autoenc.pth?download=true")
    if not os.path.exists(convnext_xl_ckpt_path):
        print(f"Downloading ConvNeXt-XL checkpoint...")
        os.system(f"wget -O {convnext_xl_ckpt_path} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/convnext_xlarge_alpha0.75_fullckpt.pth?download=true")


def init_models(device):
    """Initialize OpenCLIP ViT-bigG-14, SD VAE, and ConvNeXt-XL."""
    # OpenCLIP ViT-bigG-14 for clip_bigG_embeddings
    clip_bigG_model, _, clip_bigG_preprocess = open_clip.create_model_and_transforms(
        'ViT-bigG-14', pretrained='laion2b_s39b_b160k',
        cache_dir='datasets/transformers_cache'
    )
    clip_bigG_model = clip_bigG_model.to(device).eval()
    clip_bigG_model.visual.output_tokens = True

    # Stable Diffusion VAE for vae_latents
    vae_model = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    vae_model.load_state_dict(torch.load(vae_encode_ckpt_path, map_location=device))
    vae_model.requires_grad_(False)
    vae_model.eval()
    vae_model.to(device)

    # ConvNeXt-XL for cnx_features
    cnx_model = ConvnextXL(path=convnext_xl_ckpt_path)
    cnx_model.requires_grad_(False)
    cnx_model.eval()
    cnx_model.to(device)

    return clip_bigG_model, clip_bigG_preprocess, vae_model, cnx_model


def generate_embeddings(input_path, output_path):
    """Generate embeddings for all images in the input DataFrame."""
    device = f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu'

    print(f"Loading images DataFrame from {input_path}...")
    images_df = pd.read_pickle(input_path)
    print(f"Loaded {len(images_df)} images.")

    print("Initializing models...")
    clip_bigG_model, clip_bigG_preprocess, vae_model, cnx_model = init_models(device)

    # Pre-compute normalization constants for ConvNeXt
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(3, 1, 1)

    emb_records = []
    for _, row in tqdm(images_df.iterrows(), total=len(images_df), desc="Generating embeddings"):
        image_index = int(row['image_index'])
        image_numpy = row['image_data']  # (H, W, 3) uint8

        # Prepare image tensors
        image_pil = Image.fromarray(image_numpy)
        image_tensor_float = (
            torch.tensor(image_numpy.transpose(2, 0, 1)).float() / 255.0
        ).unsqueeze(0).to(device)  # (1, 3, H, W), range [0, 1]

        with torch.no_grad():
            # OpenCLIP ViT-bigG-14 token embeddings (256 tokens, 1664 dim)
            clip_bigG_input = clip_bigG_preprocess(image_pil).unsqueeze(0).to(device)
            _, clip_bigG_tokens = clip_bigG_model.visual.forward(clip_bigG_input)
            clip_bigG_emb = clip_bigG_tokens.cpu().squeeze(0).numpy()

            # Stable Diffusion VAE latent encoding
            vae_latent = vae_model.encode(2 * image_tensor_float - 1).latent_dist.mode().cpu().squeeze(0).numpy() * 0.18215

            # ConvNeXt-XL features
            cnx_input = (image_tensor_float - mean) / std
            cnx_feat = cnx_model.forward(cnx_input)[1].cpu().squeeze(0).numpy()

        # Shape assertions
        assert clip_bigG_emb.shape == (256, 1664), f"Unexpected clip_bigG shape: {clip_bigG_emb.shape}"
        assert vae_latent.shape == (4, 28, 28), f"Unexpected vae_latent shape: {vae_latent.shape}"
        assert cnx_feat.shape == (49, 512), f"Unexpected cnx_features shape: {cnx_feat.shape}"

        emb_records.append({
            'image_index': image_index,
            'clip_bigG_embeddings': clip_bigG_emb,
            'vae_latents': vae_latent,
            'cnx_features': cnx_feat,
        })

    emb_df = pd.DataFrame(emb_records)
    emb_df.to_pickle(output_path)
    print(f"Saved {len(emb_df)} embedding records to {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate embeddings from an images DataFrame pkl.")
    parser.add_argument(
        '--input', type=str, default='datasets/processed/things_images_df.pkl',
        help='Path to the input images DataFrame pkl file.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to the output embeddings pkl file. Defaults to <input_name>_emb.pkl.'
    )
    args = parser.parse_args()

    # Derive output path if not specified: nsd_images_df.pkl -> nsd_images_emb.pkl
    if args.output is None:
        base_dir = os.path.dirname(args.input)
        base_name = os.path.basename(args.input)
        # Replace _df.pkl with _emb.pkl, or append _emb if pattern doesn't match
        if base_name.endswith('_df.pkl'):
            out_name = base_name.replace('_df.pkl', '_emb.pkl')
        else:
            out_name = base_name.replace('.pkl', '_emb.pkl')
        args.output = os.path.join(base_dir, out_name)

    download_checkpoints()
    generate_embeddings(args.input, args.output)
