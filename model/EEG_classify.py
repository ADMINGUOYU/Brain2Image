# EEG Classification model
# Classifies EEG signals into K-means cluster categories (default 5 classes)
# Supports CBraMod and ATMS backbones

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .encoders.cbramod_eeg_encoder import CBraMod_EEG_Encoder
from .encoders.atms_eeg_encoder import ATMS_EEG_Encoder

import typing


class EEG_Classify(nn.Module):
    """
    EEG_Classify

    Pipeline:
        eeg_input -> eeg_encoder (backbone) -> flatten -> MLP classifier -> (B, num_classes)

    CBraMod path: (B, 63, 1, 200) -> flatten (B, 12600) -> MLP -> (B, num_classes)
    ATMS path:    (B, proj_dim)                           -> MLP -> (B, num_classes)
    """

    def __init__(self, param: typing.Dict):
        """
        Args:
            param['EEG_Encoder']: encoder config dict (encoder_type, cuda, etc.)
            param['Classifier']: classifier config (num_classes, hidden_dim, dropout, mlp_layers)
        """
        super().__init__()

        if 'EEG_Encoder' not in param:
            raise ValueError("Missing 'EEG_Encoder' configuration in param.")

        enc_cfg = param['EEG_Encoder']
        cls_cfg = param.get('Classifier', {})
        self.encoder_type = enc_cfg.get('encoder_type', 'CBraMod')

        # ── Backbone ─────────────────────────────────────────────────────────
        if self.encoder_type == 'CBraMod':
            self.eeg_encoder = CBraMod_EEG_Encoder(enc_cfg)
        elif self.encoder_type == 'ATMS':
            self.eeg_encoder = ATMS_EEG_Encoder(enc_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

        # ── Input dimension ──────────────────────────────────────────────────
        if self.encoder_type == 'CBraMod':
            self.flatten = Rearrange('b c s d -> b (c s d)')
            in_dim = 63 * 1 * 200  # 12600
        elif self.encoder_type == 'ATMS':
            in_dim = enc_cfg.get('proj_dim', 1024)

        # ── Classifier MLP head ──────────────────────────────────────────────
        self.num_classes = cls_cfg.get('num_classes', 5)
        hidden_dim = cls_cfg.get('hidden_dim', 512)
        dropout = cls_cfg.get('dropout', 0.3)
        mlp_layers = cls_cfg.get('mlp_layers', 2)

        layers = []
        current_dim = in_dim
        for _ in range(mlp_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: (B, 63, 1, 200) or (B, 63, 1, 250) for ATMS
        Returns:
            logits: (B, num_classes)
        """
        feats = self.eeg_encoder(eeg_input)
        if self.encoder_type == 'CBraMod':
            feats = self.flatten(feats)  # (B, 12600)
        # ATMS already outputs (B, proj_dim)
        return self.classifier(feats)

    def calc_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)

    def save_model(self, path: str):
        encoder_sd = self.eeg_encoder.state_dict()
        head_sd = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith('eeg_encoder.')
        }
        torch.save({
            'eeg_encoder': encoder_sd,
            'classifier_head': head_sd,
            'parameters': {
                'num_classes': self.num_classes,
                'EEG_Encoder_type': type(self.eeg_encoder).__name__,
            }
        }, path)

    def load_model(self, path: str, device: torch.device):
        loaded = torch.load(path, map_location=device)
        saved_type = loaded['parameters'].get('EEG_Encoder_type', 'ERROR')
        current_type = type(self.eeg_encoder).__name__
        if saved_type != current_type:
            raise ValueError(
                f"Encoder type mismatch: saved={saved_type}, current={current_type}."
            )
        self.eeg_encoder.load_state_dict(loaded['eeg_encoder'])
        if 'classifier_head' in loaded:
            current_state = self.state_dict()
            for k, v in loaded['classifier_head'].items():
                if k in current_state:
                    current_state[k].copy_(v)
