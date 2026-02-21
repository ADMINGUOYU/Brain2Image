# EEG-fMRI alignment model
# Encoders are pure backbones; all projection/pooling heads live here.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .encoders.cbramod_eeg_encoder import CBraMod_EEG_Encoder
from .encoders.atms_eeg_encoder import ATMS_EEG_Encoder
from .layers import DynamicAttentionPooling, MultiTokenViT

import typing


class EEG_fMRI_Align(nn.Module):
    """
    EEG_fMRI_Align

    Pipeline:
        eeg_input -> eeg_encoder (backbone) -> projection head -> F.normalize -> (B, out_dim)

    Projection head is built here based on encoder_type × pooling_type:
        CBraMod + flatten        : Rearrange + MLP
        CBraMod + attention      : reshape + DynamicAttentionPooling + MLP
        CBraMod + multitoken_vit : reshape + MultiTokenViT
        ATMS                     : optional Linear(proj_dim→out_mlp_dim) + GELU + Linear
    """

    def __init__(self, param: typing.Dict):
        """
        Args:
            param['EEG_Encoder']: encoder config dict (encoder_type, pooling_type, etc.)
            param['Loss']:        loss hyperparameters
        """
        super(EEG_fMRI_Align, self).__init__()

        if 'EEG_Encoder' not in param:
            raise ValueError("Missing 'EEG_Encoder' configuration in param.")
        if 'Loss' not in param:
            print("Warning: 'Loss' configuration not found in param. Using defaults.")

        enc_cfg = param['EEG_Encoder']
        self.encoder_type = enc_cfg.get('encoder_type', 'CBraMod')

        # ── Backbone ──────────────────────────────────────────────────────────
        if self.encoder_type == 'CBraMod':
            self.eeg_encoder = CBraMod_EEG_Encoder(enc_cfg)
        elif self.encoder_type == 'ATMS':
            self.eeg_encoder = ATMS_EEG_Encoder(enc_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

        # ── Projection head ───────────────────────────────────────────────────
        if self.encoder_type == 'CBraMod':
            self.pooling_type = enc_cfg.get('pooling_type', 'flatten')
            out_dim = enc_cfg.get('embedding_dim', 4096)

            if self.pooling_type == 'flatten':
                layers = [Rearrange('b c s d -> b (c s d)')]
                in_dim = 63 * 1 * 200  # 12600
                for _ in range(enc_cfg.get('mlp_layers', 2) - 1):
                    layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
                    in_dim = out_dim
                layers.append(nn.Linear(in_dim, out_dim))
                self.head = nn.Sequential(*layers)

            elif self.pooling_type == 'attention':
                attention_heads = enc_cfg.get('attention_heads', 8)
                self.reshape = Rearrange('b c s d -> b c (s d)')
                self.attention_pool = DynamicAttentionPooling(
                    num_heads=attention_heads, num_channels=63, feature_dim=200
                )
                mlp_layers = []
                in_dim = attention_heads * 200
                for _ in range(enc_cfg.get('mlp_layers', 2) - 1):
                    mlp_layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
                    in_dim = out_dim
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                self.head = nn.Sequential(*mlp_layers)

            elif self.pooling_type == 'multitoken_vit':
                self.reshape = Rearrange('b c s d -> b c (s d)')
                self.multitoken_vit = MultiTokenViT(
                    num_tokens=enc_cfg.get('num_tokens', 4),
                    num_channels=63,
                    feature_dim=200,
                    nhead=enc_cfg.get('num_attention_heads', 4),
                    num_layers=enc_cfg.get('num_transformer_layers', 4)
                )
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        elif self.encoder_type == 'ATMS':
            proj_dim = enc_cfg.get('proj_dim', 1024)
            out_mlp_dim = enc_cfg.get('out_mlp_dim', None)
            if out_mlp_dim is not None:
                self.output_mlp = nn.Sequential(
                    nn.Linear(proj_dim, out_mlp_dim),
                    nn.GELU()
                )
            else:
                self.output_mlp = None

        # ── Loss hyperparameters ──────────────────────────────────────────────
        self.mse_scale = param.get('Loss', {}).get('mse_scale', 1.0)
        self.infonce_scale = param.get('Loss', {}).get('infonce_scale', 1.0)
        self.proto_distill_scale = param.get('Loss', {}).get('proto_distill_scale', 1.0)
        self.temperature = param.get('Loss', {}).get('temperature', 0.07)
        self.normalize_fmri = param.get('Loss', {}).get('normalize_fmri', True)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: (B, 63, 1, 200)
        Returns:
            eeg_embeds: (B, out_dim), unit-normalized
        """
        feats = self.eeg_encoder(eeg_input)

        if self.encoder_type == 'CBraMod':
            if self.pooling_type == 'flatten':
                eeg_embeds = self.head(feats)
            elif self.pooling_type == 'attention':
                feats = self.reshape(feats)          # (B, 63, 200)
                pooled = self.attention_pool(feats)  # (B, K*200)
                eeg_embeds = self.head(pooled)
            elif self.pooling_type == 'multitoken_vit':
                feats = self.reshape(feats)          # (B, 63, 200)
                eeg_embeds = self.multitoken_vit(feats)

        elif self.encoder_type == 'ATMS':
            eeg_embeds = feats  # (B, proj_dim)
            if self.output_mlp is not None:
                eeg_embeds = self.output_mlp(eeg_embeds)

        eeg_embeds = F.normalize(eeg_embeds, dim=-1)
        return eeg_embeds  # (B, out_dim)

    def save_model(self, path: str):
        eeg_encoder_state_dict = self.eeg_encoder.state_dict()

        # Projection head = everything that is NOT the eeg_encoder
        projection_head_state_dict = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith('eeg_encoder.')
        }

        parameters = {
            'mse_scale': self.mse_scale,
            'infonce_scale': self.infonce_scale,
            'proto_distill_scale': self.proto_distill_scale,
            'temperature': self.temperature,
            'normalize_fmri': self.normalize_fmri,
            'EEG_Encoder_type': type(self.eeg_encoder).__name__
        }

        torch.save({
            'eeg_encoder': eeg_encoder_state_dict,
            'projection_head': projection_head_state_dict,
            'parameters': parameters
        }, path)

    def load_model(self, path: str, device: torch.device):
        loaded = torch.load(path, map_location=device)
        saved_encoder_type = loaded['parameters'].get('EEG_Encoder_type', 'ERROR')
        current_encoder_type = type(self.eeg_encoder).__name__
        if saved_encoder_type != current_encoder_type:
            raise ValueError(
                f"Encoder type mismatch: saved={saved_encoder_type}, current={current_encoder_type}."
            )
        self.eeg_encoder.load_state_dict(loaded['eeg_encoder'])
        if 'projection_head' in loaded:
            current_state = self.state_dict()
            for k, v in loaded['projection_head'].items():
                if k in current_state:
                    current_state[k].copy_(v)
        self.mse_scale = loaded['parameters']['mse_scale']
        self.infonce_scale = loaded['parameters']['infonce_scale']
        self.proto_distill_scale = loaded['parameters']['proto_distill_scale']
        self.temperature = loaded['parameters']['temperature']
        self.normalize_fmri = loaded['parameters']['normalize_fmri']

    def calc_alignment_loss(self,
                            aligned_embeds: torch.Tensor,
                            target_embeds: torch.Tensor,
                            label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aligned_embeds: (B, out_dim) — unit-normalized EEG embeddings
            target_embeds:  (B, out_dim) — fMRI embeddings
            label:          (B,)         — cluster labels
        Returns:
            total_loss, mse_loss, infonce_loss, proto_loss
        """
        if self.normalize_fmri:
            target_embeds = F.normalize(target_embeds, dim=1)

        mse_loss = F.mse_loss(aligned_embeds, target_embeds, reduction='none').sum(dim=1).mean()

        logits = torch.matmul(aligned_embeds, target_embeds.T) / self.temperature
        labels = torch.arange(aligned_embeds.size(0), device=aligned_embeds.device)
        infonce_loss = F.cross_entropy(logits, labels)

        exist_classes = torch.unique(label)
        n_cls = exist_classes.shape[0]
        emb_dim = aligned_embeds.shape[1]
        device = aligned_embeds.device

        eeg_proto  = torch.zeros(n_cls, emb_dim, device=device)
        fmri_proto = torch.zeros(n_cls, emb_dim, device=device)

        for idx, cls in enumerate(exist_classes):
            mask = (label == cls)
            eeg_proto[idx]  = aligned_embeds[mask].mean(dim=0)
            fmri_proto[idx] = target_embeds[mask].mean(dim=0)

        eeg_proto  = F.normalize(eeg_proto,  dim=1)
        fmri_proto = F.normalize(fmri_proto, dim=1)
        proto_loss = F.mse_loss(eeg_proto, fmri_proto, reduction='none').sum(dim=1).mean()

        total_loss = (self.mse_scale * mse_loss
                      + self.infonce_scale * infonce_loss
                      + self.proto_distill_scale * proto_loss)
        return total_loss, mse_loss, infonce_loss, proto_loss

    def get_metrics_for_alignment(self,
                                  aligned_embeds: torch.Tensor,
                                  target_embeds: torch.Tensor) -> typing.Tuple[float, float, float, float]:
        """
        Returns: mse, cos_sim, retrieval_acc_top1, retrieval_acc_top10
        """
        if self.normalize_fmri:
            target_embeds = F.normalize(target_embeds, dim=1)

        mse = F.mse_loss(aligned_embeds, target_embeds, reduction='none').sum(dim=1).mean().item()
        cos_sim = F.cosine_similarity(aligned_embeds, target_embeds, dim=1).mean().item()

        pred_norm   = F.normalize(aligned_embeds, dim=1)
        target_norm = F.normalize(target_embeds,  dim=1)
        sim_matrix  = torch.matmul(pred_norm, target_norm.T)
        labels      = torch.arange(len(aligned_embeds), device=aligned_embeds.device)

        top_1 = torch.argmax(sim_matrix, dim=1)
        retrieval_acc = (top_1 == labels).float().mean().item()

        k = min(10, len(aligned_embeds))
        top_k_indices = torch.topk(sim_matrix, k=k, dim=1).indices
        retrieval_acc_top10 = (top_k_indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        return mse, cos_sim, retrieval_acc, retrieval_acc_top10


# Test code
if __name__ == "__main__":
    batch_size = 4

    # ── CBraMod / multitoken_vit ──────────────────────────────────────────────
    param_cbramod = {
        'EEG_Encoder': {
            'encoder_type': 'CBraMod',
            'pooling_type': 'multitoken_vit',
            'num_tokens': 4,
            'num_transformer_layers': 4,
            'num_attention_heads': 4,
            'use_pretrained_weights': False
        },
        'Loss': {
            'mse_scale': 1.0, 'infonce_scale': 1.0,
            'proto_distill_scale': 1.0, 'temperature': 0.07,
            'normalize_fmri': True
        }
    }
    model_cbramod = EEG_fMRI_Align(param_cbramod)
    dummy_eeg = torch.randn(batch_size, 63, 1, 200)
    out = model_cbramod(dummy_eeg)
    print("CBraMod output shape:", out.shape)  # (4, 4096)

    # ── ATMS ─────────────────────────────────────────────────────────────────
    param_atms = {
        'EEG_Encoder': {
            'encoder_type': 'ATMS',
            'num_channels': 63, 'seq_len': 250,
            'emb_size': 40, 'proj_dim': 1024,
            'drop_proj': 0.5, 'd_model': 250,
            'n_heads': 4, 'e_layers': 1,
            'd_ff': 256, 'dropout': 0.25, 'factor': 1,
            'out_mlp_dim': 4096,
            'use_pretrained_weights': False
        },
        'Loss': {
            'mse_scale': 1.0, 'infonce_scale': 1.0,
            'proto_distill_scale': 1.0, 'temperature': 0.07,
            'normalize_fmri': True
        }
    }
    model_atms = EEG_fMRI_Align(param_atms)
    dummy_eeg_atms = torch.randn(batch_size, 63, 1, 250)
    out_atms = model_atms(dummy_eeg_atms)
    print("ATMS output shape:", out_atms.shape)  # (4, 4096)
