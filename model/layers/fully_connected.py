import torch

class FullyConnectedLayer(torch.nn.Module):

    """
    Fully connected layer
    Args:
        input_size: The dimensionality of the input features.
        output_size: The dimensionality of the output features.
        dropout: The dropout rate to apply to the output features.
        normalization: An optional normalization layer to apply after the linear transformation.
        activation: An optional activation function to apply after the linear transformation.
    """

    def __init__(self, input_size, output_size,
                 dropout = 0.0, 
                 normalization: torch.nn.Module = None,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        
        super(FullyConnectedLayer, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        self.normalization = normalization
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear.forward(x)
        if self.normalization:
            x = self.normalization.forward(x)
        if self.dropout:
            x = self.dropout.forward(x)
        x = self.activation.forward(x)
        return x

# Test the implementation
if __name__ == "__main__":
    batch_size = 2
    input_size = 5
    output_size = 3

    # Create a random input tensor
    x = torch.rand(batch_size, input_size)

    # Initialize the fully connected layer
    fc_layer = FullyConnectedLayer(input_size, output_size, dropout = 0.2,
                                   normalization = torch.nn.BatchNorm1d(output_size),
                                   activation = torch.nn.ReLU())

    # Compute the output
    output = fc_layer.forward(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)