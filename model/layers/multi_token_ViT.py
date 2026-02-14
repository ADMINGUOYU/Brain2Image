import torch
import torch.nn as nn

class MultiTokenViT(nn.Module):

    """
    Multi-Token Vision Transformer (ViT) with learnable tokens and positional embeddings.
    This module implements a vision transformer that uses learnable tokens to represent global features
    and positional embeddings to encode spatial information. The transformer processes the concatenated
    sequence of learnable tokens and input channels, then extracts the learned tokens for downstream tasks.

    MLP bottle neck: 2048
    MLP output: 4096
    """

    def __init__(self, 
                 num_tokens = 4, 
                 num_channels = 63, feature_dim = 200, 
                 nhead = 4, num_layers = 4):

        """
        Args:
            num_tokens (int): Number of learnable tokens to use.
            num_channels (int): Number of input channels (e.g., 63).
            feature_dim (int): Dimensionality of the feature space (e.g., 200).
            nhead (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder layers.
        """

        super().__init__()
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # Learnable tokens
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, feature_dim) * 0.02)

        # Positional embeddings for total sequence length
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + num_channels, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = feature_dim,
            nhead = nhead,
            dim_feedforward = feature_dim * 4,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        # MLP head
        mlp_input_dim = num_tokens * feature_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (batch, num_channels, feature_dim)
        batch_size = x.size(0)

        # Expand learnable tokens to batch size
        tokens = self.tokens.expand(batch_size, -1, -1)  # (batch, num_tokens, feature_dim)

        # Concatenate tokens with input
        x = torch.cat([tokens, x], dim = 1)  # (batch, num_tokens + num_channels, feature_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Pass through transformer
        x = self.transformer.forward(x)  # (batch, num_tokens + num_channels, feature_dim)

        # Extract first num_tokens
        x = x[:, :self.num_tokens, :]  # (batch, num_tokens, feature_dim)

        # Flatten
        x = x.reshape(batch_size, -1)  # (batch, num_tokens * feature_dim)

        # Pass through MLP
        out = self.mlp.forward(x)  # (batch, 4096)

        return out