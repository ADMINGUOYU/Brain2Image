import torch
import typing

class ScaledDotProductAttention(torch.nn.Module):

    """
    Scaled dot product attention module.
    Args:
        dropout: The dropout rate to apply to the attention weights.
    """
    def __init__(self, dropout = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask = None
                ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Compute the scaled dot product attention.
        Args:
            query: The input query tensor of shape (batch_size, head_count, seq_len, d_model).
            key: The input key tensor of shape (batch_size, head_count, seq_len, d_model).
            value: The input value tensor of shape (batch_size, head_count, seq_len, d_model).
            mask: An optional mask tensor to prevent attention to certain positions.
            (mask shape: (batch_size, seq_len) or (batch_size, seq_len, seq_len))
        Returns:
            Tuple of (output, attn_weights):
             - output: shape (batch_size, head_count, seq_len, d_model)
             - attn_weights: shape (batch_size, head_count, seq_len, seq_len)
        """
        
        # Calculate the attention scores
        d_k = query.size(-1)
        scores = torch.matmul(query, 
                              key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))

        # Apply the mask (if provided) to the attention scores
        if mask is not None:
            # shape of scores: (batch_size, head_count, seq_len, seq_len)
            # shape of mask: (batch_size, seq_len) or (batch_size, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            else:
                raise ValueError("Mask must be of shape (batch_size, seq_len) or (batch_size, seq_len, seq_len)")
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get the attention weights and apply dropout
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the output by multiplying the attention weights with the value tensor
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(torch.nn.Module):

    """
    MultiHeadAttention
    Args:
        d_model: The dimensionality of the input and output features.
        num_heads: The number of attention heads.
        dropout: The dropout rate to apply to the attention weights.
    """
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        # Ensure that the model dimension is divisible by the number of heads
        assert d_model % num_heads == 0

        # Calculate the dimension of each head
        self.head_dimension = d_model // num_heads

        # Store the number of heads and the model dimension
        self.num_heads = num_heads
        self.d_model = d_model

        # Define linear layers for query, key, value, and output projections
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_out = torch.nn.Linear(d_model, d_model)

        # Initialize the scaled dot product attention module
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask = None,
                ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Compute the multi-head attention output.
        Args:
            query: The input query tensor of shape (batch_size, seq_len, d_model).
            key: The input key tensor of shape (batch_size, seq_len, d_model).
            value: The input value tensor of shape (batch_size, seq_len, d_model).
            mask: An optional mask tensor to prevent attention to certain positions.
        Returns:
            Tuple of (output, attn_weights):
            - The output tensor of shape (batch_size, seq_len, d_model) after applying multi-head attention.
            - The attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        
        # Get the batch size from the query tensor
        batch_size = query.size(0)

        # Linear projections
        # shape after linear layer: (batch_size, seq_len, d_model)
        # shape after view and transpose: (batch_size, num_heads, seq_len, head_dimension)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        
        # Apply attention
        # shape of attn_output: (batch_size, num_heads, seq_len, head_dimension)
        # shape of attn_weights: (batch_size, num_heads, seq_len, seq_len)
        attn_output, attn_weights = self.attention.forward(query, key, value, mask)

        # Concatenate heads and apply final linear layer
        # shape of attn_output after transpose: (batch_size, seq_len, num_heads, head_dimension)
        # shape after view: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dimension)
        output = self.linear_out(attn_output)

        return output, attn_weights

# Test the implementation
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_model = 9
    num_heads = 3

    # Create random input tensors for query, key, and value
    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)

    # Create a mask tensor (mask half of the positions)
    mask = torch.zeros(batch_size, seq_len).bool()
    mask[:, :seq_len//2] = True  # keep the first half of the positions, mask the second half

    # Initialize the multi-head attention module
    multi_head_attention = MultiHeadAttention(d_model, num_heads)

    # Compute the multi-head attention output
    output, attn_weights = multi_head_attention(query, key, value, mask)

    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)

    print("Output:", output)
    print("Attention weights:", attn_weights)