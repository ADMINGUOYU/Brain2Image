"""
EEG-VAR Generation Model

Stage 2 VAR-based generation model that combines:
1. Frozen stage 1 EEG-CLIP alignment model (produces 1024-dim embeddings)
2. Frozen VQVAE (multi-scale image tokenization)
3. Trainable VAR transformer (autoregressive image generation)

Critical fixes implemented:
- Per-scale loss weighting to prevent high-resolution scale dominance
- VAR's internal embed_proj maps CLIP embeddings to condition space
- CFG training with 10% dropout on EEG embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from model.EEG_CLIP_align import EEG_CLIP_Align
from model.var.vqvae import VQVAE
from model.var.var_eeg import VAREEG


class EEG_VAR_Generation(nn.Module):
    """
    Stage 2 VAR-based generation model.
    Combines frozen stage 1 EEG-CLIP alignment with trainable VAR.
    """

    def __init__(
        self,
        stage1_config: dict,
        stage1_ckpt_path: str,
        vae_ckpt_path: str,
        var_ckpt_path: str,
        var_config: dict,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device

        # 1. Load frozen stage 1 model (EEG-CLIP alignment)
        print(f"Loading stage 1 model from {stage1_ckpt_path}")
        self.eeg_clip_model = EEG_CLIP_Align(stage1_config)
        self.eeg_clip_model.load_model(stage1_ckpt_path, device)
        self.eeg_clip_model.eval()
        for p in self.eeg_clip_model.parameters():
            p.requires_grad = False
        print("Stage 1 model loaded and frozen")

        # 2. Load frozen VQVAE
        print(f"Loading VQVAE from {vae_ckpt_path}")
        self.vae = VQVAE(
            vocab_size=var_config['vocab_size'],
            z_channels=var_config['z_channels'],
            ch=var_config['ch'],
            test_mode=True,
            share_quant_resi=4,
            v_patch_nums=var_config['patch_nums'],
        )
        vae_state = torch.load(vae_ckpt_path, map_location=device)
        self.vae.load_state_dict(vae_state, strict=True)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        print("VQVAE loaded and frozen")

        # 3. Load pre-trained VAR (trainable)
        # NOTE: VAR already has embed_proj (Linear 1024 -> 1024) inside
        # This is sufficient for mapping CLIP embeddings to VAR condition space
        print(f"Loading VAR from {var_ckpt_path}")
        self.var = VAREEG(
            vae_local=self.vae,
            depth=var_config['depth'],
            embed_dim=var_config['embed_dim'],
            num_heads=var_config['num_heads'],
            mlp_ratio=var_config['mlp_ratio'],
            drop_rate=var_config.get('drop_rate', 0.0),
            attn_drop_rate=var_config.get('attn_drop_rate', 0.0),
            drop_path_rate=var_config.get('drop_path_rate', 0.0),
            norm_eps=var_config.get('norm_eps', 1e-6),
            shared_aln=var_config.get('shared_aln', False),
            cond_drop_rate=var_config.get('cond_drop_rate', 0.1),
            attn_l2_norm=var_config.get('attn_l2_norm', False),
            patch_nums=var_config['patch_nums'],
            flash_if_available=var_config.get('flash_if_available', True),
            fused_if_available=var_config.get('fused_if_available', True),
        )
        var_state = torch.load(var_ckpt_path, map_location=device)
        # Load with strict=False to allow missing keys (e.g., class_emb if not used)
        self.var.load_state_dict(var_state, strict=False)
        print("VAR loaded (trainable)")

        # 4. Per-scale loss weighting (uniform by default)
        # Prevents high-resolution scales from dominating gradients
        patch_nums = var_config['patch_nums']  # (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        L = sum(pn ** 2 for pn in patch_nums)  # 685
        self.register_buffer('loss_weight', torch.ones(1, L) / L)
        print(f"Per-scale loss weighting initialized (L={L})")

    def forward(self, eeg: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            eeg: EEG input (B, 63, 1, 200/250)
            images: Ground truth images (B, 3, 256, 256) in [-1, 1]

        Returns:
            logits_BLV: VAR logits (B, 685, 4096)
            gt_idx_Bl: Ground truth token indices (list of tensors)
        """
        # Get frozen EEG embeddings (L2-normalized CLIP-aligned)
        with torch.no_grad():
            eeg_emb = self.eeg_clip_model(eeg)  # (B, 1024), unit-normalized

        # Online VAE tokenization
        with torch.no_grad():
            gt_idx_Bl = self.vae.img_to_idxBl(images)
            x_BLCv = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        # VAR forward (teacher forcing)
        # VAR's embed_proj will map eeg_emb to condition space
        logits_BLV = self.var(x_BLCv, eeg_emb)
        return logits_BLV, gt_idx_Bl

    def compute_loss(self, logits_BLV: torch.Tensor, gt_idx_Bl: List[torch.Tensor], label_smoothing: float = 0.1) -> torch.Tensor:
        """
        Compute per-scale weighted cross-entropy loss.

        Args:
            logits_BLV: VAR logits (B, 685, 4096)
            gt_idx_Bl: Ground truth token indices (list of tensors)
            label_smoothing: Label smoothing factor

        Returns:
            loss: Scalar loss value
        """
        B, L, V = logits_BLV.shape
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # (B, 685)

        # Compute per-token loss (reduction='none')
        loss_BL = F.cross_entropy(
            logits_BLV.view(-1, V),
            gt_BL.view(-1),
            label_smoothing=label_smoothing,
            reduction='none'
        ).view(B, L)  # (B, 685)

        # Apply per-scale weighting (uniform: 1/685 per token)
        # This ensures each scale contributes equally to the gradient
        loss = loss_BL.mul(self.loss_weight).sum(dim=-1).mean()
        return loss

    def compute_per_scale_metrics(
        self,
        logits_BLV: torch.Tensor,
        gt_idx_Bl: List[torch.Tensor],
        label_smoothing: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        Compute loss and accuracy for each scale separately.

        Args:
            logits_BLV: VAR logits (B, 680, 4096)
            gt_idx_Bl: Ground truth token indices (list of 10 tensors)
            label_smoothing: Label smoothing factor

        Returns:
            per_scale_losses: List of 10 scalars (loss per scale)
            per_scale_accs: List of 10 scalars (accuracy per scale)
        """
        B, L, V = logits_BLV.shape
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # (B, 680)

        # Compute per-token loss
        loss_BL = F.cross_entropy(
            logits_BLV.view(-1, V),
            gt_BL.view(-1),
            label_smoothing=label_smoothing,
            reduction='none'
        ).view(B, L)  # (B, 680)

        # Compute per-token accuracy
        pred_BL = torch.argmax(logits_BLV, dim=-1)  # (B, 680)
        correct_BL = (pred_BL == gt_BL).float()  # (B, 680)

        # Extract per-scale metrics using pre-computed boundaries
        per_scale_losses = []
        per_scale_accs = []

        for begin, end in self.var.begin_ends:
            # Loss: average over batch and tokens in this scale
            scale_loss = loss_BL[:, begin:end].mean()
            per_scale_losses.append(scale_loss.item())

            # Accuracy: average over batch and tokens in this scale
            scale_acc = correct_BL[:, begin:end].mean()
            per_scale_accs.append(scale_acc.item())

        return per_scale_losses, per_scale_accs

    def compute_topk_accuracy(
        self,
        logits_BLV: torch.Tensor,
        gt_idx_Bl: List[torch.Tensor],
        k: int = 5
    ) -> float:
        """
        Compute top-k accuracy (whether correct token is in top-k predictions).

        Args:
            logits_BLV: VAR logits (B, 680, 4096)
            gt_idx_Bl: Ground truth token indices
            k: Top-k value

        Returns:
            topk_acc: Top-k accuracy
        """
        B, L, V = logits_BLV.shape
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # (B, 680)

        # Get top-k predictions
        topk_preds = torch.topk(logits_BLV, k=k, dim=-1).indices  # (B, 680, k)

        # Check if GT is in top-k
        gt_expanded = gt_BL.unsqueeze(-1)  # (B, 680, 1)
        correct = (topk_preds == gt_expanded).any(dim=-1).float()  # (B, 680)

        return correct.mean().item()

    def compute_perplexity(
        self,
        logits_BLV: torch.Tensor,
        gt_idx_Bl: List[torch.Tensor]
    ) -> float:
        """
        Compute perplexity = exp(cross_entropy).
        Lower perplexity indicates better model confidence.

        Args:
            logits_BLV: VAR logits (B, 680, 4096)
            gt_idx_Bl: Ground truth token indices

        Returns:
            perplexity: Perplexity value
        """
        loss = self.compute_loss(logits_BLV, gt_idx_Bl, label_smoothing=0.0)
        return torch.exp(loss).item()

    def compute_prediction_entropy(
        self,
        logits_BLV: torch.Tensor
    ) -> float:
        """
        Compute normalized prediction entropy (range [0, 1]).

        - Value near 0: Model is confident (potential mode collapse)
        - Value near 1: Model is uncertain (random guessing)

        Args:
            logits_BLV: VAR logits (B, 680, 4096)

        Returns:
            normalized_entropy: Entropy normalized by log(vocab_size)
        """
        import math

        V = logits_BLV.size(-1)
        probs = F.softmax(logits_BLV, dim=-1)  # (B, 680, 4096)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, 680)

        # Normalize by maximum possible entropy: log(V)
        max_entropy = math.log(V)
        normalized_entropy = entropy.mean().item() / max_entropy

        return normalized_entropy

    def generate(
        self,
        eeg: torch.Tensor,
        cfg_scale: float = 1.5,
        top_k: int = 900,
        top_p: float = 0.95,
        g_seed: int = None
    ) -> torch.Tensor:
        """
        Autoregressive image generation with classifier-free guidance.

        Args:
            eeg: EEG input (B, 63, 1, 200/250)
            cfg_scale: Classifier-free guidance scale
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            g_seed: Random seed for generation

        Returns:
            images: Generated images (B, 3, 256, 256) in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            # Get EEG embeddings (L2-normalized CLIP-aligned)
            eeg_emb = self.eeg_clip_model(eeg)  # (B, 1024)

            # Autoregressive generation with CFG
            # VAR's autoregressive_infer_cfg handles the null condition internally
            images = self.var.autoregressive_infer_cfg(
                B=eeg.shape[0],
                eeg_emb=eeg_emb,
                cfg=cfg_scale,
                top_k=top_k,
                top_p=top_p,
                g_seed=g_seed
            )
        return images  # (B, 3, 256, 256) in [0, 1]
