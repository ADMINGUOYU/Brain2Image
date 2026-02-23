# EEG-fMRI End-to-End generation model
# Combines alignment pipeline (EEG encoder + projection → 4096-dim)
# with MindEye2's BrainNetwork and diffusion prior for image reconstruction.
#
# Pipeline:
#   EEG → EEG_fMRI_Align (encoder + projection) → 4096-dim embedding
#       → BrainNetwork (MLP Mixer) → backbone (B, 256, 1664)
#       → BrainDiffusionPrior → diffusion loss on CLIP patch tokens
#       + blurry reconstruction branch → VAE latents + ConvNeXt features
#
# Loss = align_scale * alignment_loss
#       + prior_scale * diffusion_prior_loss
#       + clip_loss_scale * clip_contrastive_loss
#       + blur_scale * blurry_reconstruction_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from .EEG_fMRI_align import EEG_fMRI_Align
from .MindEYE2 import (
    BrainNetwork,
    BrainDiffusionPrior,
    PriorNetwork,
    soft_clip_loss,
    soft_cont_loss,
    mixco_nce,
    cosine_anneal,
)

import typing


class EEG_fMRI_E2E(nn.Module):
    """
    End-to-End EEG-to-Image generation model.

    Wraps:
      1. EEG_fMRI_Align — EEG encoder + projection head + alignment loss
      2. BrainNetwork — MLP Mixer backbone → (B, 256, 1664) + blurry branch
      3. BrainDiffusionPrior — diffusion prior on CLIP patch tokens
    """

    def __init__(self, param: typing.Dict):
        """
        Args:
            param['EEG_Encoder']: encoder config dict
            param['Loss']:        alignment loss hyperparameters
            param['Generation']:  generation pipeline config
        """
        super().__init__()

        gen_cfg = param.get('Generation', {})

        # ── 1. Alignment sub-model (EEG encoder + projection + alignment loss) ──
        self.align_model = EEG_fMRI_Align(param)

        # ── 2. BrainNetwork (MLP Mixer backbone) ──
        clip_size = gen_cfg.get('clip_size', 1664)
        blurry_recon = gen_cfg.get('blurry_recon', True)
        clip_scale = gen_cfg.get('clip_scale', 1.0)
        n_blocks = gen_cfg.get('n_blocks', 4)
        drop = gen_cfg.get('drop', 0.15)

        self.brain_network = BrainNetwork(
            h=4096,
            in_dim=4096,
            out_dim=clip_size * 256,
            seq_len=1,
            n_blocks=n_blocks,
            drop=drop,
            clip_size=clip_size,
            blurry_recon=blurry_recon,
            clip_scale=clip_scale,
        )

        # ── 3. Diffusion Prior ──
        out_dim = clip_size
        depth = 6
        dim_head = 52
        heads = out_dim // dim_head

        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=256,
            learned_query_mode="pos_emb",
        )

        self.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        # ── Loss scale hyperparameters ──
        self.align_scale = gen_cfg.get('align_scale', 1.0)
        self.prior_scale = gen_cfg.get('prior_scale', 30.0)
        self.clip_loss_scale = gen_cfg.get('clip_loss_scale', 1.0)
        self.blur_scale = gen_cfg.get('blur_scale', 0.5)
        self.blurry_recon = blurry_recon
        self.clip_size = clip_size

        # -- Generation (MindEYE2) ckpt loading --
        mindeye2_ckpt_path = gen_cfg.get('mindeye2_ckpt_path', None)
        if mindeye2_ckpt_path is not None:
            print(f"Loading MindEYE2 weights from {mindeye2_ckpt_path} into BrainNetwork and BrainDiffusionPrior...")
            # we use 'cpu' for now
            # when we move the model to GPU later, the weights will be on the same device as the rest of the model
            self.brain_network.load_mindeye2_weights(mindeye2_ckpt_path, 'cpu')
            self.diffusion_prior.load_mindeye2_weights(mindeye2_ckpt_path, 'cpu')

    def forward_encoder(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEG encoder only (before MixCo).
        Args:
            eeg_input: (B, 63, 1, T)
        Returns:
            eeg_embeds: (B, 4096), unit-normalized
        """
        return self.align_model.forward(eeg_input)

    def forward_generation(self, eeg_embeds: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        """
        Forward pass from 4096-dim embeddings through BrainNetwork + blurry branch.
        Called AFTER MixCo is applied to eeg_embeds.
        Args:
            eeg_embeds: (B, 4096) — possibly mixed embeddings
        Returns:
            dict with backbone, clip_voxels, blurry_latents, blurry_features
        """
        eeg_3d = eeg_embeds.unsqueeze(1)  # (B, 1, 4096)
        backbone, clip_voxels, blurry = self.brain_network(eeg_3d)

        result = {
            'backbone': backbone,       # (B, 256, 1664)
            'clip_voxels': clip_voxels,  # (B, 256, 1664)
        }

        if self.blurry_recon and isinstance(blurry, tuple):
            result['blurry_latents'] = blurry[0]   # (B, 4, 28, 28)
            result['blurry_features'] = blurry[1]   # (B, 49, 512)
        else:
            result['blurry_latents'] = None
            result['blurry_features'] = None

        return result

    def calc_e2e_loss(
        self,
        eeg_embeds: torch.Tensor,
        fmri: torch.Tensor,
        label: torch.Tensor,
        gen_outputs: typing.Dict[str, torch.Tensor],
        clip_target: torch.Tensor,
        vae_latents: torch.Tensor,
        cnx_features: torch.Tensor,
        epoch: int,
        num_epochs: int,
        perm: torch.Tensor = None,
        betas: torch.Tensor = None,
        select: torch.Tensor = None,
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Compute all E2E losses.
        Args:
            eeg_embeds:   (B, 4096) — the ORIGINAL (unmixed) embeddings for alignment loss
            fmri:         (B, 4096) — fMRI targets
            label:        (B,) — cluster labels
            gen_outputs:  dict from forward_generation (using mixed embeddings)
            clip_target:  (B, 256, 1664) — pre-computed ViT-bigG patch tokens
            vae_latents:  (B, 4, 28, 28) — pre-computed SD VAE latents
            cnx_features: (B, 49, 512) — pre-computed ConvNeXt features
            epoch, num_epochs: for temperature annealing
            perm, betas, select: MixCo parameters (None if no MixCo)
        Returns:
            dict of losses: total, align, prior, clip, blur, mse, infonce, proto
        """
        # 1. Alignment loss (on original unmixed embeddings)
        align_total, mse_loss, infonce_loss, proto_loss = \
            self.align_model.calc_alignment_loss(eeg_embeds, fmri, label)

        # 2. Diffusion prior loss
        backbone = gen_outputs['backbone']  # (B, 256, 1664)
        prior_loss, _ = self.diffusion_prior(
            text_embed=backbone,
            image_embed=clip_target,
        )

        # 3. CLIP contrastive loss
        clip_voxels = gen_outputs['clip_voxels']  # (B, 256, 1664)
        clip_voxels_flat = clip_voxels.flatten(1)  # (B, 256*1664)
        clip_target_flat = clip_target.flatten(1)   # (B, 256*1664)

        if perm is not None:
            # MixCo active — use mixco_nce
            clip_loss = mixco_nce(
                clip_voxels_flat, clip_target_flat,
                temp=0.006, perm=perm, betas=betas, select=select,
            )
        else:
            # No MixCo — use soft_clip_loss with annealed temperature
            epoch_temps = cosine_anneal(0.004, 0.0075, num_epochs)
            epoch_temp = epoch_temps[min(epoch, num_epochs - 1)]
            clip_loss = soft_clip_loss(clip_voxels_flat, clip_target_flat, temp=epoch_temp)

        # 4. Blurry reconstruction loss
        blur_loss = torch.tensor(0.0, device=eeg_embeds.device)
        if self.blurry_recon and gen_outputs['blurry_latents'] is not None:
            blur_loss = F.l1_loss(gen_outputs['blurry_latents'], vae_latents)
            if cnx_features is not None:
                # Debug print shapes
                # print(f"[DEBUG] Blurry features shape: {gen_outputs['blurry_features'].shape}, "
                #       f"ConvNeXt features shape: {cnx_features.shape}")
                blur_loss = blur_loss + 0.1 * soft_cont_loss(
                    gen_outputs['blurry_features'].reshape(gen_outputs['blurry_features'].size(0), -1),
                    cnx_features.reshape(cnx_features.size(0), -1),
                    cnx_features.reshape(cnx_features.size(0), -1), 
                    temp=0.2
                )

        # Total weighted loss
        total = (self.align_scale * align_total
                 + self.prior_scale * prior_loss
                 + self.clip_loss_scale * clip_loss
                 + self.blur_scale * blur_loss)

        return {
            'total': total,
            'align': align_total,
            'prior': prior_loss,
            'clip': clip_loss,
            'blur': blur_loss,
            'mse': mse_loss,
            'infonce': infonce_loss,
            'proto': proto_loss,
        }

    def save_model(self, path: str):
        align_state = self.align_model.state_dict()
        brain_network_state = self.brain_network.state_dict()
        diffusion_prior_state = self.diffusion_prior.state_dict()

        parameters = {
            'align_scale': self.align_scale,
            'prior_scale': self.prior_scale,
            'clip_loss_scale': self.clip_loss_scale,
            'blur_scale': self.blur_scale,
            'blurry_recon': self.blurry_recon,
            'clip_size': self.clip_size,
            'EEG_Encoder_type': type(self.align_model.eeg_encoder).__name__,
        }

        torch.save({
            'align_model': align_state,
            'brain_network': brain_network_state,
            'diffusion_prior': diffusion_prior_state,
            'parameters': parameters,
        }, path)

    def load_model(self, path: str, device: torch.device):
        loaded = torch.load(path, map_location=device)
        saved_encoder_type = loaded['parameters'].get('EEG_Encoder_type', 'ERROR')
        current_encoder_type = type(self.align_model.eeg_encoder).__name__
        if saved_encoder_type != current_encoder_type:
            raise ValueError(
                f"Encoder type mismatch: saved={saved_encoder_type}, current={current_encoder_type}."
            )
        self.align_model.load_state_dict(loaded['align_model'])
        self.brain_network.load_state_dict(loaded['brain_network'])
        self.diffusion_prior.load_state_dict(loaded['diffusion_prior'])
        self.align_scale = loaded['parameters']['align_scale']
        self.prior_scale = loaded['parameters']['prior_scale']
        self.clip_loss_scale = loaded['parameters']['clip_loss_scale']
        self.blur_scale = loaded['parameters']['blur_scale']

    def load_alignment_checkpoint(self, path: str, device: torch.device):
        """Load a pretrained alignment model checkpoint into self.align_model."""
        self.align_model.load_model(path, device)
        print(f"Loaded alignment checkpoint from {path}")


# Test code
if __name__ == "__main__":
    batch_size = 4

    param = {
        'EEG_Encoder': {
            'encoder_type': 'CBraMod',
            'pooling_type': 'multitoken_vit',
            'num_tokens': 4,
            'num_transformer_layers': 4,
            'num_attention_heads': 4,
            'use_pretrained_weights': False,
        },
        'Loss': {
            'mse_scale': 1.0, 'infonce_scale': 1.0,
            'proto_distill_scale': 1.0, 'temperature': 0.07,
            'normalize_fmri': True,
        },
        'Generation': {
            'n_blocks': 4, 'drop': 0.15, 'clip_size': 1664,
            'blurry_recon': True, 'clip_scale': 1.0,
            'prior_scale': 30.0, 'clip_loss_scale': 1.0,
            'blur_scale': 0.5, 'align_scale': 1.0,
        },
    }

    model = EEG_fMRI_E2E(param)
    dummy_eeg = torch.randn(batch_size, 63, 1, 200)

    # Stage 1: encoder
    eeg_embeds = model.forward_encoder(dummy_eeg)
    print("EEG embeds shape:", eeg_embeds.shape)  # (4, 4096)

    # Stage 2: generation
    gen_out = model.forward_generation(eeg_embeds)
    print("Backbone shape:", gen_out['backbone'].shape)  # (4, 256, 1664)
    print("CLIP voxels shape:", gen_out['clip_voxels'].shape)  # (4, 256, 1664)
    print("Blurry latents shape:", gen_out['blurry_latents'].shape)  # (4, 4, 28, 28)
    print("Blurry features shape:", gen_out['blurry_features'].shape)  # (4, 49, 512)

    # Loss computation
    dummy_fmri = torch.randn(batch_size, 4096)
    dummy_label = torch.randint(0, 5, (batch_size,))
    dummy_clip_target = torch.randn(batch_size, 256, 1664)
    dummy_vae = torch.randn(batch_size, 4, 28, 28)
    dummy_cnx = torch.randn(batch_size, 49, 512)

    losses = model.calc_e2e_loss(
        eeg_embeds, dummy_fmri, dummy_label, gen_out,
        dummy_clip_target, dummy_vae, dummy_cnx,
        epoch=0, num_epochs=100,
    )
    print("Total loss:", losses['total'].item())
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
