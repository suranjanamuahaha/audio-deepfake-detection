import torch
import torch.nn as nn

class MaxFeatureMap(nn.Module):

    def forward(self, x):

        shape = x.shape

        x = x.view(
            shape[0],
            shape[1] // 2,
            2,
            shape[2]
        )

        x, _ = torch.max(x, dim=2)

        return x


class LCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(768, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            MaxFeatureMap()
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.net(x)

        x = x.transpose(1, 2)

        return x