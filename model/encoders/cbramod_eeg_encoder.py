import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ..CBraMod import CBraMod
from ..layers import DynamicAttentionPooling, MultiTokenViT

import typing

class CBraMod_EEG_Encoder(CBraMod):

    """
    CBraMod_EEG_Encoder
    """

    def __init__(self, param: typing.Dict):
        super(CBraMod_EEG_Encoder, self).__init__()
        """
        Initialize CBraMod_EEG_Encoder model.
        Args:
            param: Configuration object containing model parameters.
            - param['cuda']: CUDA device index to load pretrained weights (if use_pretrained_weights is True)
            - param['use_pretrained_weights']: Whether to load pretrained weights (default False)
            - param['foundation_dir']: Path to pretrained weights (if use_pretrained_weights is True)

            - param['pooling_type']: Type of pooling to use ('flatten', 'attention', 'multitoken_vit') (default 'flatten')

            - param['attention_heads']: Number of attention heads for 'attention' pooling (default 8)

            - param['num_tokens']: Number of tokens for 'multitoken_vit' pooling (default 4)
            - param['num_transformer_layers']: Number of transformer layers for 'multitoken_vit' pooling (default 4)
            - param['num_attention_heads']: Number of attention heads for 'multitoken_vit' pooling (default 4)

            - param['mlp_layers']: Number of layers in the MLP head for 'flatten' and 'attention' pooling (default 2)
            - param['embedding_dim']: Embedding dimension for MLP layers for 'flatten' and 'attention' pooling (default 512)
        """

        # Backbone
        self.backbone = CBraMod(
            in_dim = 200, out_dim = 200, d_model = 200,
            dim_feedforward = 800, seq_len = 30,
            n_layer = 12, nhead = 8
        )

        # Load pretrained weights
        if param.get('use_pretrained_weights', False):
            map_location = torch.device(f'cuda:{param['cuda']}') if torch.cuda.is_available() else 'cpu'
            self.backbone.load_state_dict(
                torch.load(param['foundation_dir'], map_location = map_location)
            )

        # Replace output projection with identity
        self.backbone.proj_out = nn.Identity()
    
        # Get pooling configuration
        self.pooling_type = param.get('pooling_type', 'flatten')

        # Build head based on pooling type
        if self.pooling_type == 'flatten':
            # Original: Flatten + MLP
            layers = [Rearrange('b c s d -> b (c s d)')]
            in_dim = 63 * 1 * 200  # 12600

            for i in range(param['mlp_layers'] - 1):
                layers.extend([
                    nn.Linear(in_dim, param['embedding_dim']),
                    nn.GELU()
                ])
                in_dim = param['embedding_dim']

            layers.append(nn.Linear(in_dim, param['embedding_dim']))
            self.head = nn.Sequential(*layers)

            # Output dimension
            self.output_dim = param['embedding_dim']

        elif self.pooling_type == 'attention':

            # Number of attention heads
            self.attention_heads = param.get('attention_heads', 8)

            # Reshape to sequence: (batch, 63, 1, 200) -> (batch, 63, 200)
            self.reshape = Rearrange('b c s d -> b c (s d)')

            # Dynamic attention pooling with K heads
            self.attention_pool = DynamicAttentionPooling(
                num_heads=self.attention_heads,
                num_channels=63,
                feature_dim=200
            )

            # MLP after pooling
            # Input dimension is now K * 200 instead of embedding_dim
            mlp_layers = []
            in_dim = self.attention_heads * 200  # K * 200
            for i in range(param['mlp_layers'] - 1):
                mlp_layers.extend([
                    nn.Linear(in_dim, param['embedding_dim']),
                    nn.GELU()
                ])
                in_dim = param['embedding_dim']
            mlp_layers.append(nn.Linear(in_dim, param['embedding_dim']))

            self.head = nn.Sequential(*mlp_layers)

            # Output dimension
            self.output_dim = param['embedding_dim']

        elif self.pooling_type == 'multitoken_vit':
            # Reshape to sequence: (batch, 63, 1, 200) -> (batch, 63, 200)
            self.reshape = Rearrange('b c s d -> b c (s d)')

            # Get configuration
            num_tokens = param.get('num_tokens', 4)
            num_transformer_layers = param.get('num_transformer_layers', 4)
            num_attention_heads = param.get('num_attention_heads', 4)

            # Multi-Token ViT pooling (outputs 4096 directly)
            self.multitoken_vit = MultiTokenViT(
                num_tokens=num_tokens,
                num_channels=63,
                feature_dim=200,
                nhead=num_attention_heads,
                num_layers=num_transformer_layers
            )

            # No additional MLP needed - MultiTokenViT outputs 4096 directly
            self.head = nn.Identity()

            # Output dimension
            self.output_dim = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (batch, 63, 1, 200)

        if self.pooling_type == 'flatten':
            out = self.head(feats)
        elif self.pooling_type == 'attention':
            # Reshape
            feats = self.reshape(feats)  # (batch, 63, 200)

            # Weighted average pooling
            pooled = self.attention_pool(feats)  # (batch, K * 200)

            # MLP
            out = self.head(pooled)

        elif self.pooling_type == 'multitoken_vit':
            # Reshape
            feats = self.reshape(feats)  # (batch, 63, 200)

            # Multi-Token ViT pooling
            out = self.multitoken_vit(feats)  # (batch, 4096)

        return out

    # WARNING: to be removed (we add fMRI thing in different module)
    # def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) \
    #     -> torch.Tensor:

    #     # MSE loss
    #     mse_loss = F.mse_loss(pred, target)

    #     # Scale MSE by 1000 if fMRI is normalized to match InfoNCE range
    #     if self.normalize_fmri:
    #         mse_loss = mse_loss * 1000

    #     # InfoNCE loss
    #     pred_norm = F.normalize(pred, dim=1)
    #     target_norm = F.normalize(target, dim=1)
    #     logits = torch.matmul(pred_norm, target_norm.T) / self.temperature
    #     labels = torch.arange(pred.size(0), device=pred.device)
    #     infonce_loss = F.cross_entropy(logits, labels)

    #     # Combined loss
    #     total_loss = self.mse_scale * mse_loss + self.infonce_scale * infonce_loss

    #     return total_loss, mse_loss, infonce_loss

# Test code
if __name__ == "__main__":
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 63, 1, 200)

    # Create dummy param object
    param_vit = {
        'pooling_type': 'multitoken_vit',  # 'flatten', 'attention', or 'multitoken_vit'
        'num_tokens': 4,
        'num_transformer_layers': 4,
        'num_attention_heads': 4,
        'use_pretrained_weights': False
    }

    param_flatten = {
        'pooling_type': 'flatten',  # 'flatten', 'attention', or 'multitoken_vit'
        'mlp_layers': 2,
        'embedding_dim': 512,
        'use_pretrained_weights': False
    }

    param_attention = {
        'pooling_type': 'attention',  # 'flatten', 'attention', or 'multitoken_vit'
        'attention_heads': 8,
        'mlp_layers': 2,
        'embedding_dim': 512,
        'use_pretrained_weights': False
    }

    # Create model instance
    model = CBraMod_EEG_Encoder(param_vit)

    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)
