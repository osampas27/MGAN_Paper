import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.5):
        """
        Multi-head Graph Attention Layer.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            GATLayer(input_dim, output_dim, dropout) for _ in range(num_heads)
        ])

    def forward(self, x, adj):
        outputs = [head(x, adj) for head in self.attention_heads]
        return torch.cat(outputs, dim=-1)  # Concatenation of multi-head outputs

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(output_dim, num_heads=1, dropout=dropout)

    def forward(self, x, adj):
        attn_output, _ = self.attn(x, x, x)
        return F.elu(attn_output)  # Apply ELU activation
