import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x)


# ✅ LAZY LOAD PART
_model = None

def get_model():
    global _model

    if _model is None:
        print("🔥 Loading model...")

        _model = DeepfakeDetector()
        _model.load_state_dict(
            torch.load("model.pth", map_location="cpu")
        )
        _model.eval()

    return _model