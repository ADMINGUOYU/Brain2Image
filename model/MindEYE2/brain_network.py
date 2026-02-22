# BrainNetwork MLP Mixer backbone from MindEyeV2
# Reference: 
# https://github.com/MedARC-AI/MindEyeV2/blob/main/src/models.py
# BrainNetwork class - no major changes

import numpy as np
import torch
import torch.nn as nn

# Import from diffusers
# Please make sure to install diffusers library
from diffusers.models.autoencoders.vae import Decoder

class BrainNetwork(nn.Module):

    """
    MindEYE2 Brain Network MLP Mixer backbone.
    It handles:
     - ouput 1664 x 256 features for CLIP image reconstruction
     - blurry reconstructions of 28x28 for auxiliary loss (low level submodule)
    """

    def __init__(self, 
                 h: int = 4096, 
                 in_dim: int = 4096, 
                 out_dim: int = 1664 * 256, 
                 seq_len: int = 1,
                 n_blocks: int = 4, 
                 drop: float = .15, 
                 clip_size: int = 1664, 
                 blurry_recon: bool = True, 
                 clip_scale: float = 1.0):

        """
        Args:
            h: hidden dimension of the MLP mixer blocks (same as in_dim)
            in_dim: input dimension of the MLP mixer blocks (input dimension of BrainNetwork)
            out_dim: output dimension of the backbone linear layer (output shape flattened -> clip_emb_dim(1664) * clip_seq_dim(256))
            seq_len: sequence length (number of tokens) ???
            n_blocks: number of MLP mixer blocks
            drop: dropout rate for the MLP mixer blocks
            clip_size: output dimension of the CLIP projector (clip_emb_dim(1664))
            blurry_recon: whether to perform blurry reconstructions for auxiliary loss
            clip_scale: scaling factor for the CLIP features (if 0, no CLIP features are outputted)
        """
        super().__init__()

        # Store parameters (init)
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale

        # MLP mixer blocks
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias = True) 
        self.clip_proj = self.projector(clip_size, clip_size, h = clip_size)
        
        # Blurry reconstructions for auxiliary loss (low level submodule)
        if self.blurry_recon:
            # Pass MLP
            self.blin1 = nn.Linear(h * seq_len, 4 * 28 * 28,bias = True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            # Upsampler (decoder) for blurry reconstructions
            self.bupsampler = Decoder(
                in_channels = 64,
                out_channels = 4,
                up_block_types = ["UpDecoderBlock2D",\
                                  "UpDecoderBlock2D",\
                                  "UpDecoderBlock2D"],
                block_out_channels = [32, 64, 128],
                layers_per_block = 1,
            )
            # Projector for blurry reconstructions 
            # (to get blurry feature maps for auxiliary loss)
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias = False),
                nn.GroupNorm(1, 512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias = False),
                nn.GroupNorm(1, 512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias = True),
            )
            
    def projector(self, in_dim: int, out_dim: int, h: int = 2048)\
        -> nn.Sequential:
        """
        Args:
            in_dim: input dimension of the projector (clip_emb_dim(1664))
            out_dim: output dimension of the projector (clip_emb_dim(1664))
            h: hidden dimension of the projector
        """
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim: int, out_dim: int, drop: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h: int, drop: float) -> nn.Sequential:
        """
        Token mixing
        """
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),
        )

    def mixer_block2(self, seq_len: int, drop: float) -> nn.Sequential:
        """
        Channel mixing
        """
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)
        )
        
    def forward(self, x: torch.Tensor) \
        -> (tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):

        """
        Args:
            x: input tensor of shape (batch_size, seq_len, in_dim)
        Returns:
            backbone: output tensor of shape (batch_size, clip_seq_dim(256), clip_emb_dim(1664))
            c: output tensor of shape (batch_size, clip_seq_dim(256), clip_emb_dim(1664)) (CLIP features for CLIP image reconstruction)
            b: tuple of output tensors of shape 
               (batch_size, 4, 28, 28) (blurry reconstructions for auxiliary loss)
               and
               (batch_size, 49, 512) (projected from the blurry reconstructions) (blurry feature maps for auxiliary loss)
        """

        # make empty tensors
        # (clip + blur OPTIONALS)
        c, b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
        # Mixer blocks
        residual1 = x                   # (batch_size, seq_len, in_dim)
        residual2 = x.permute(0, 2, 1)  # (batch_size, in_dim, seq_len)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1.forward(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)
            
            x = block2.forward(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)
        
        # Output linear layer
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear.forward(x).reshape(len(x), -1, self.clip_size)
        
        # CLIP features for CLIP image reconstruction
        if self.clip_scale > 0:
            c = self.clip_proj.forward(backbone)

        # Blurry reconstructions for auxiliary loss (low level submodule)
        if self.blurry_recon:
            b = self.blin1.forward(x)
            b = self.bdropout.forward(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm.forward(b)
            b_aux = self.b_maps_projector.forward(b).flatten(2).permute(0, 2, 1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler.forward(b), b_aux)
        
        return backbone, c, b

# Testing code
if __name__ == "__main__":
    # Create a random input tensor of shape (batch_size, seq_len, in_dim)
    batch_size = 2
    seq_len = 1
    in_dim = 4096
    x = torch.randn(batch_size, seq_len, in_dim)

    # Create an instance of the BrainNetwork
    model = BrainNetwork()

    # Forward pass
    backbone, c, b = model.forward(x)

    # Print output shapes
    print("Backbone output shape:", backbone.shape)  # Expected: (batch_size, clip_seq_dim(256), clip_emb_dim(1664))
    print("CLIP features shape:", c.shape)           # Expected: (batch_size, clip_seq_dim(256), clip_emb_dim(1664))
    print("Blurry reconstructions shape:", b[0].shape)  # Expected: (batch_size, 4, 28, 28)
    print("Blurry feature maps shape:", b[1].shape)     # Expected: (batch_size, 49, 512)

    # Print out
    # Backbone output shape: torch.Size([2, 256, 1664])
    # CLIP features shape: torch.Size([2, 256, 1664])
    # Blurry reconstructions shape: torch.Size([2, 4, 28, 28])
    # Blurry feature maps shape: torch.Size([2, 49, 512])