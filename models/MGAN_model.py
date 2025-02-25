import torch
import torch.nn as nn
from models.graph_attention import GraphAttention
from models.transformer_encoder import TransformerEncoder

class MGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.5):
        """
        MGAN Model - Combines Graph Attention Networks (GATs) and Transformer-based encoders.

        Args:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Hidden dimension for GAT and Transformer layers.
            output_dim (int): Number of output classes (e.g., 2 for binary classification).
            num_heads (int): Number of attention heads for GAT.
            num_layers (int): Number of Transformer layers.
            dropout (float): Dropout rate.
        """
        super(MGAN, self).__init__()

        self.graph_attention = GraphAttention(input_dim, hidden_dim, num_heads, dropout)
        self.transformer_encoder = TransformerEncoder(hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Final classification layer

    def forward(self, x, adj):
        """
        Forward pass through MGAN.

        Args:
            x (Tensor): Input features (batch_size, num_nodes, feature_dim).
            adj (Tensor): Adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Predicted class probabilities.
        """
        x = self.graph_attention(x, adj)  # Graph-based aggregation
        x = self.transformer_encoder(x)   # Temporal modeling
        output = self.fc(x.mean(dim=1))   # Aggregate sequence output
        return output
