import os
import torch
import librosa
import torch.nn.functional as F

from transformers import (
    WavLMModel,
    AutoFeatureExtractor
)

from app.model import DeepfakeDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading WavLM...")

wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-base-plus"
).to(device)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# wavlm_path = os.path.join(
#     BASE_DIR,
#     "saved_models",
#     "wavlm.pt"
# )

# wavlm.load_state_dict(
#     torch.load(
#         wavlm_path,
#         map_location=device
#     )
# )

wavlm.eval()

print("Loading classifier...")

model_path = os.path.join(
    BASE_DIR,
    "saved_models",
    "best_model.pt"
)

model = DeepfakeDetector()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)
model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus"
)

print("Model Ready")


def predict(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    max_len = 32000

    if len(audio) > max_len:

        audio = audio[:max_len]

    else:

        audio = torch.nn.functional.pad(
            torch.tensor(audio),
            (
                0,
                max_len - len(audio)
            )
        ).numpy()

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        features = wavlm(
            **inputs
        ).last_hidden_state

        output = model(
            features
        )

    probs = F.softmax(
        output,
        dim=1
    )

    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = torch.max(
        probs
    ).item()

    result = (
        "deepfake"
        if pred == 1
        else "real"
    )

    return {
        "result": result,
        "confidence": round(
            confidence,
            4
        )
    }


if __name__ == "__main__":

    test_path = os.path.join(
        BASE_DIR,
        "test.wav"
    )

    print(
        predict(
            test_path
        )
    )