import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attn = self.attention(x)

        weights = torch.softmax(attn, dim=1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled