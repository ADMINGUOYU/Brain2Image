import torch
import torch.nn as nn
import typing

from ..CBraMod import CBraMod


class CBraMod_EEG_Encoder(CBraMod):
    """
    CBraMod_EEG_Encoder — pure backbone wrapper.
    forward(x) -> (B, 63, 1, 200)

    All projection/pooling heads live in EEG_fMRI_Align.
    """

    def __init__(self, param: typing.Dict):
        super(CBraMod_EEG_Encoder, self).__init__()
        """
        Args:
            param: Configuration dict.
            - param['cuda']: CUDA device index for pretrained weight loading
            - param['use_pretrained_weights']: Whether to load pretrained weights (default False)
            - param['foundation_dir']: Path to pretrained weights
        """

        # Backbone
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        # Load pretrained weights
        if param.get('use_pretrained_weights', False):
            map_location = torch.device(f'cuda:{param["cuda"]}') if torch.cuda.is_available() else 'cpu'
            self.backbone.load_state_dict(
                torch.load(param['foundation_dir'], map_location=map_location)
            )

        # Replace output projection with identity
        self.backbone.proj_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 63, 1, 200)


# Test code
if __name__ == "__main__":
    batch_size = 4
    dummy_input = torch.randn(batch_size, 63, 1, 200)

    param = {'use_pretrained_weights': False}
    model = CBraMod_EEG_Encoder(param)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (4, 63, 1, 200)
