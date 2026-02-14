import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

class DynamicAttentionPooling(nn.Module):

    """
    Dynamic Attention Pooling Layer
    This layer implements a dynamic attention pooling mechanism that learns to focus on the most relevant channels
    for each head. It uses learnable query vectors to compute attention scores and produces a fixed-size output regardless of the input channel dimension. The output is a concatenation of the weighted averages for each head, resulting in a feature vector of size (num_heads * feature_dim). This allows the model to capture complex interactions between channels while reducing the dimensionality of the input data.
    Args:
        num_heads (int): The number of attention heads (K).
        num_channels (int): The number of input channels (C).
        feature_dim (int): The dimensionality of the feature space (D).
    Input:
        x (Tensor): Input tensor of shape (batch_size, num_channels, feature_dim).
    Output:
        out (Tensor): Output tensor of shape (batch_size, num_heads * feature_dim).
    """

    def __init__(self, num_heads, num_channels = 63, feature_dim = 200):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # K learnable query vectors (one per head)
        # Shape: (num_heads, feature_dim)
        self.queries = nn.Parameter(torch.randn(num_heads, feature_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, num_channels, feature_dim)
        batch_size = x.size(0)

        # Compute attention scores for each head
        # queries: (num_heads, feature_dim)
        # x: (batch, num_channels, feature_dim)
        # scores: (batch, num_heads, num_channels)
        scores = torch.einsum('hd,bcd->bhc', self.queries, x)

        # Apply softmax to get attention weights
        # weights: (batch, num_heads, num_channels)
        weights = F.softmax(scores, dim=2)

        # Compute weighted average for each head
        # weights: (batch, num_heads, num_channels)
        # x: (batch, num_channels, feature_dim)
        # output: (batch, num_heads, feature_dim)
        output = torch.einsum('bhc,bcd->bhd', weights, x)

        # Flatten: (batch, num_heads * feature_dim)
        out = output.reshape(batch_size, -1)

        return out  # (batch, K * feature_dim)

# Test code
if __name__ == "__main__":
    batch_size = 4
    num_channels = 63
    feature_dim = 200
    num_heads = 8

    x = torch.randn(batch_size, num_channels, feature_dim)
    pooling_layer = DynamicAttentionPooling(num_heads, num_channels, feature_dim)
    out = pooling_layer(x)
    print(out.shape)  # Expected: (batch_size, num_heads * feature_dim) -> (4, 1600)