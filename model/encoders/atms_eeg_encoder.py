"""
This script containes - ATMS EEG Encoder
 - Adapted from: dongyangli-del/EEG_Image_decode/Retrieval/ATMS_retrieval.py 

Architecture:
    Raw EEG (batch_size, n_channels, seq_len)
    NOTE: we will squeeze CBraMod style 
          (batch, n_channels, time_step, seq_len) to the above format
      -> iTransformer   [value and positional encoding]  (inverted-variable transformer, encodes each channel as a token)
      -> PatchEmbedding (ShallowConvNet-style spatial/temporal conv)
      -> FlattenHead
      -> Proj_eeg       (Linear + Residual GELU MLP + LayerNorm)
    Output: (batch, proj_dim)

    * Subject-wise linear layers are REMOVED
"""

# Import necessary libraries
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor

# ----------- Start of iTransformer module ----------- #

# Masking
class _TriangularCausalMask:
    def __init__(self, B: int, L: int, device = "cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask

# Attention
class _FullAttention(nn.Module):
    """
    Vanilla scaled-dot-product attention (no masking used in encoder).
    """

    def __init__(self, mask_flag: bool = False, factor: int = 5, scale: float = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        
        """
        Args:
            mask_flag : bool -> whether to apply causal mask (not used in encoder)
            factor : int     -> attention factor for ProbSparse attention (not used in encoder)
            scale : float    -> scaling factor for attention scores (default: 1/sqrt(d_k))
            attention_dropout : float -> dropout rate for attention weights
            output_attention : bool   -> whether to return attention weights (for visualization/debugging)
        """
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: Tensor, 
                keys: Tensor, values: Tensor, 
                attn_mask: _TriangularCausalMask = None, 
                key_padding_mask: Tensor = None) \
        -> typing.Tuple[Tensor, Tensor]:

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or (1.0 / math.sqrt(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is None:
            attn_mask = _TriangularCausalMask(B, L, device = queries.device)

        if attn_mask is not None:
            if attn_mask.mask is not None:
                scores.masked_fill_(attn_mask.mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None

class _AttentionLayer(nn.Module):
    def __init__(self, attention: _FullAttention,
                 d_model: int, n_heads: int, 
                 d_keys: int = None, d_values: int = None):

        """
        Args:
            attention : _FullAttention -> the attention mechanism to use (e.g., full attention)
            d_model : int -> the dimensionality of the input and output feature space
            n_heads : int -> the number of attention heads
            d_keys : int -> the dimensionality of the keys and queries for each head (default: d_model // n_heads)
            d_values : int -> the dimensionality of the values for each head (default: d_model // n_heads)
        """
        super().__init__()

        # Calculate if not specified
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, 
                queries: Tensor, 
                keys: Tensor, 
                values: Tensor, 
                attn_mask: _TriangularCausalMask = None, 
                key_padding_mask: Tensor = None) \
        -> typing.Tuple[Tensor, Tensor]:

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, 
            key_padding_mask = key_padding_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# Encoder Layer
# adapted from dongyangli-del/EEG_Image_decode/models/subject_layers/Transformer_EncDec.py
class _EncoderLayer(nn.Module):
    def __init__(self, 
                attention: _AttentionLayer, 
                d_model: int, 
                d_ff: int = None, 
                dropout: float = 0.1, 
                activation: str = "gelu"):
        
        """
        Args:
            attention : _AttentionLayer -> the attention layer to use
            d_model : int -> the dimensionality of the input and output feature space
            d_ff : int -> the dimensionality of the hidden layer in the feedforward network (default: 4 * d_model)
            dropout : float -> the dropout rate (default: 0.1)
            activation : str -> the activation function to use ('relu' or 'gelu') (default: "gelu")
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size = 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size = 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: Tensor, 
                attn_mask: _TriangularCausalMask = None, 
                key_padding_mask: Tensor = None) \
        -> typing.Tuple[Tensor, Tensor]:

        new_x, attn = self.attention.forward \
            (x, x, x, attn_mask = attn_mask,
            key_padding_mask = key_padding_mask)
        x = x + self.dropout.forward(new_x)
        y = x = self.norm1.forward(x)
        y = self.dropout.forward(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout.forward(self.conv2(y).transpose(-1, 1))
        return self.norm2.forward(x + y), attn

class _Encoder(nn.Module):

    # NOTE: ATMS doesn't have Conv layers, we remove it here

    def __init__(self, attn_layers: list, norm_layer: nn.Module = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: Tensor, 
                attn_mask: _TriangularCausalMask = None, 
                key_padding_mask: Tensor = None) \
        -> typing.Tuple[Tensor, list]:

        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask = attn_mask,
                                 key_padding_mask = key_padding_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

# Embedding Layers (module)
class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(_PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, :x.size(1)]
    
class _DataEmbedding(nn.Module):
    """
    Inverted embedding: projects the "time" dimension (seq_len) into d_model
    so that each channel becomes one token of length d_model.
    NOTE: 
    No temporal or subject-specific embedding (joint_train = False).
    We removed all unused parts
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        # Value embedding
        self.value_embedding = nn.Linear(seq_len, d_model)

        # Positional embedding
        self.position_embedding = _PositionalEmbedding(d_model = d_model)

        # Drop out
        self.dropout = nn.Dropout(dropout)

        # Mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: (batch, n_channels, seq_len)
        x = self.value_embedding(x)   # (batch, n_channels, d_model)

        # Always add positional embedding (important for EEG sequence modeling)
        x = x + self.position_embedding.forward(x)

        # Add masking
        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        return self.dropout(x)

class _iTransformer(nn.Module):
    """
    Inverted-variable Transformer.
    Input : (batch, n_channels, seq_len)
    Output: (batch, n_channels, d_model)   — only first n_channels tokens kept
    """

    def __init__(self, seq_len: int, d_model: int, n_heads: int,
                 e_layers: int, d_ff: int, dropout: float, factor: int,
                 n_channels: int = 63, output_attention: bool = False):
        
        """
        Args:
            seq_len : int -> the length of the input sequence (number of time points)
            d_model : int -> the dimensionality of the output feature space for each token
            n_heads : int -> the number of attention heads in the multi-head attention mechanism
            e_layers : int -> the number of encoder layers to stack
            d_ff : int -> the dimensionality of the hidden layer in the feedforward network within each encoder layer
            dropout : float -> the dropout rate to apply in the attention and feedforward layers
            factor : int -> the attention factor for ProbSparse attention (not used in encoder)
            n_channels : int -> the number of input channels (tokens) to keep in the output
            output_attention : bool -> whether to return attention weights (for visualization/debugging)
        """
        super().__init__()
        self.output_attention = output_attention
        self.n_channels = n_channels

        self.enc_embedding = _DataEmbedding(seq_len, d_model, dropout)

        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    _AttentionLayer(
                        _FullAttention(
                            False, factor,
                            attention_dropout = dropout,
                            output_attention = output_attention
                        ),
                        d_model, n_heads
                    ),
                    d_model, 
                    d_ff,
                    dropout = dropout,
                    activation = "gelu"
                )
                for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_channels, seq_len)
        enc_out = self.enc_embedding.forward(x)                          # (B, C, d_model)
        enc_out, _ = self.encoder.forward(enc_out, attn_mask = None)
        enc_out = enc_out[:, :self.n_channels, :]                # (B, C, d_model)
        return enc_out

# ------------ END of iTransformer module ------------ #

# -- Start of PatchEmbedding / ResidualAdd modules --- #

class _PatchEmbedding(nn.Module):
    """
    Adapted from ShallowNet.
    Input : (batch, n_channels, d_model)  — treated as (B, C, T) image
    Output: (batch, n_patches, emb_size)
    """

    def __init__(self, n_channels: int = 63, emb_size: int = 40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride = (1, 1)),
            nn.AvgPool2d((1, 51),  (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1), stride = (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride = (1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_channels, d_model) -> add channel dim ->  (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)   # (B, n_patches, emb_size)
        return x

class _ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)

# --- END of PatchEmbedding / ResidualAdd modules ---- #

# ---------------------------------------------------- #
#                   ATMS EEG Encoder                   #
# ---------------------------------------------------- #

class ATMS_EEG_Encoder(nn.Module):
    """
    ATMS EEG Encoder
    - Adapted from: dongyangli-del/EEG_Image_decode/Retrieval/ATMS_retrieval.py 
    
    Flow of encoding process (Pipeline):
    EEG (batch_size, n_channels, seq_len)
      -> iTransformer                           (batch_size, n_channels, d_model)
      -> PatchEmbedding [conv spatial/temporal] (batch_size, n_patches,  emb_size)
      -> Flatten                                (batch_size, n_patches * emb_size)
      -> Proj_eeg [Linear + Residual GELU + LN] (batch_size, proj_dim)

    output_dim == proj_dim
    """

    def __init__(self, param: typing.Dict):
        """
        Args:
            param : dict containing the following keys (with defaults):
                param['num_channels']      : int,   default 63
                param['seq_len']           : int,   default 250
                param['num_subjects']      : int,   default 10
                param['emb_size']          : int,   default 40    (PatchEmbedding output channels)
                param['proj_dim']          : int,   default 1024  (final embedding dim = output_dim)
                param['drop_proj']         : float, default 0.5
                param['d_model']           : int,   default 250   (iTransformer hidden dim)
                param['n_heads']           : int,   default 4
                param['e_layers']          : int,   default 1
                param['d_ff']              : int,   default 256
                param['dropout']           : float, default 0.25
                param['factor']            : int,   default 1     (attention factor)

                param['out_mlp_dim']       : int,   default None   (dimension of the output MLP head, if used)

                param['cuda']                   : int,  ckpt loading mapping device
                param['use_pretrained_weights'] : bool, whether to load pretrained weights
                param['foundation_dir']         : str,  directory to load pretrained weights
        """
        super().__init__()

        # Get parameters with defaults
        n_channels  = param.get("num_channels", 63)
        seq_len     = param.get("seq_len",      250)
        emb_size    = param.get("emb_size",     40)
        proj_dim    = param.get("proj_dim",     1024)
        drop_proj   = param.get("drop_proj",    0.5)
        d_model     = param.get("d_model",      250)
        n_heads     = param.get("n_heads",      4)
        e_layers    = param.get("e_layers",     1)
        d_ff        = param.get("d_ff",         256)
        dropout     = param.get("dropout",      0.25)
        factor      = param.get("factor",       1)
        # Initialize components

        # -> Inverted-variable Transformer
        self.backbone = _iTransformer(
            seq_len = seq_len, d_model = d_model,
            n_heads = n_heads, e_layers = e_layers,
            d_ff = d_ff, dropout = dropout, factor = factor,
            n_channels = n_channels,
        )

        # -> ShallowConvNet patch embedding
        #    input to tsconv is (B, 1, n_channels, d_model)
        #    NOTE: the spatial conv uses kernel (n_channels, 1)
        self.patch_embed = _PatchEmbedding(n_channels = n_channels, emb_size = emb_size)

        # * Compute output dimension
        with torch.no_grad():
            _dummy = torch.zeros(1, n_channels, seq_len)
            _enc   = self.backbone(_dummy)          # (1, C, d_model)
            _pat   = self.patch_embed(_enc)             # (1, n_patches, emb_size)
            embedding_dim = _pat.shape[1] * _pat.shape[2]
        print(f"[ATMS init DEBUG] Computed embedding_dim (n_patches * emb_size) = {embedding_dim}")

        # -> Projection head  (Proj_eeg)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            _ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

        # Output dimension is proj_dim
        # NOTE: this is accessible outside class
        self.output_dim = proj_dim

        # Load pretrained weights if specified
        if param.get('use_pretrained_weights', False):
            map_location = torch.device(f'cuda:{param['cuda']}') if torch.cuda.is_available() else 'cpu'
            self.load_atms_weights(param['foundation_dir'], map_location = map_location)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
        ----
        x : Tensor  shape (batch, n_channels, seq_len)
        NOTE: if input is in CBraMod format (batch, n_channels, time_step, seq_len),
              we'll help you reshape it to (batch, n_channels, seq_len) by squeezing the time_step dimension.

        Returns
        -------
        Tensor  shape (batch, output_dim)
        """
        # Check input shape
        if x.dim() == 4:
            # we concat all time steps extending the seq_len dimension
            b, c, t, s = x.shape
            x = x.view(b, c, t * s)  # (batch, n_channels, time_step * seq_len)
        elif x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, n_channels, seq_len) or (batch, n_channels, time_step, seq_len), but got {x.shape}")

        # Perform forward pass
        x = self.backbone.forward(x)       # (batch_size, n_channels, d_model)
        x = self.patch_embed.forward(x)        # (batch_size, n_patches, emb_size)
        x = x.contiguous().view(x.size(0), -1) # (batch_size, n_patches * emb_size)
        x = self.proj.forward(x)               # (batch_size, proj_dim)
        return x

    def load_atms_weights(self, weight_path: str, map_location = None):
        """
        Load pretrained weights for the ATMS.
        NOTE: this function is meant to be compatible with weights from:
            dongyangli-del/EEG_Image_decode/Retrieval/ATMS_retrieval.py 
        Args:
            weight_path : str -> path to the pretrained weights file (should be a .pth or .pt file containing the state_dict)
            map_location : torch.device or str -> device mapping for loading the weights (e.g., 'cpu' or 'cuda:0')
        """

        # Warn user about this function
        print(f"\033[91mWARNING: You are using load_atms_weights() which is designed to load pretrained weights from dongyangli-del/EEG_Image_decode/Retrieval/ATMS_retrieval.py. Make sure your weight file is compatible with this format.\033[0m")

        # Set default map_location if not provided
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the state dict
        state_dict = torch.load(weight_path, map_location = map_location)

        # Define mapping table (different start names)
        # Key: ATMS pretrained name
        # Value: current model name
        mapping_table = {
            'encoder.enc_embedding': 'backbone.enc_embedding',
            'encoder.encoder': 'backbone.encoder',
            'enc_eeg.0' : 'patch_embed',
            'proj_eeg' : 'proj',
        }

        # We now load
        print(f"Loading pretrained weights from {weight_path} to ATMS_EEG_Encoder...")
        # NOTE: We make sure shapes matches
        for pretrained_key, current_key in mapping_table.items():
            # Track loaded parameters for this key
            loaded_params = 0
            # Find all keys in state_dict that start with pretrained_key
            matching_keys = [k for k in state_dict.keys() if k.startswith(pretrained_key)]
            for key in matching_keys:
                # Get the corresponding current model key by replacing the prefix
                new_key = key.replace(pretrained_key, current_key, 1)
                if new_key in self.state_dict():
                    if self.state_dict()[new_key].shape == state_dict[key].shape:
                        self.state_dict()[new_key].copy_(state_dict[key])
                        loaded_params += 1
                    else:
                        print(f"\033[91mShape mismatch for {key} -> {new_key}: {state_dict[key].shape} vs {self.state_dict()[new_key].shape}\033[0m")
                else:
                    print(f"\033[91mKey {new_key} not found in current model state dict.\033[0m")

            # Print summary of loading
            # print in green if loaded_params == len(matching_keys), else print in yellow
            if loaded_params == len(matching_keys):
                print(f"\033[92mSuccessfully loaded all {loaded_params} parameters for {pretrained_key} -> {current_key}\033[0m")
            else:
                print(f"\033[93mLoaded {loaded_params}/{len(matching_keys)} parameters for {pretrained_key} -> {current_key}. Please check the above messages for details.\033[0m")
                
# Testing code
if __name__ == "__main__":
    param = {
        "num_channels": 63,
        "seq_len":      250,
        "emb_size":     40,
        "proj_dim":     1024,
        "drop_proj":    0.5,
        "d_model":      250,
        "n_heads":      4,
        "e_layers":     1,
        "d_ff":         256,
        "dropout":      0.25,
        "factor":       1,
    }

    # Test initialization
    encoder = ATMS_EEG_Encoder(param)
    print(f"output_dim : {encoder.output_dim}")

    # Test weight loading
    encoder.load_atms_weights("datasets/processed/atms/sub-01.pth")

    dummy = torch.randn(8, param["num_channels"], 1, param["seq_len"])  # (batch, n_channels, time_step, seq_len)
    out   = encoder.forward(dummy)
    print(f"input  shape : {dummy.shape}")    # (8, 63, 1, 250)
    print(f"output shape : {out.shape}")      # (8, 1024)