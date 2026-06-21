import torch.nn as nn

from app.lcnn import LCNN
from app.attention_pooling import AttentionPooling


class DeepfakeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.lcnn = LCNN()

        self.pool = AttentionPooling(
            input_dim=64
        )

        self.classifier = nn.Sequential(

            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.lcnn(x)

        x = self.pool(x)

        return self.classifier(x)