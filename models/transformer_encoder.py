import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.5):
        """
        Transformer Encoder for sequence modeling.

        Args:
            hidden_dim (int): Hidden dimension for Transformer layers.
            num_layers (int): Number of Transformer layers.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)  # Apply layer normalization
