import os
import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from app.model import DeepfakeDetector
from app.utils import processor

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔥 Load wav2vec with cache (prevents repeated downloads)
print("Loading wav2vec...")
wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base",
    cache_dir=os.path.join(BASE_DIR, "hf_cache")
).to(device)

# 🔥 Load classifier
print("Loading classifier...")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "model.pt")

model = DeepfakeDetector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model ready")


def predict(file_path):
    try:
        print("🎧 Loading:", file_path)

        # ✅ Load audio safely
        audio, sr = librosa.load(file_path, sr=16000)

        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio received")

        # ✅ Fix length (2 sec = 32000 samples)
        max_len = 32000

        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            padding = max_len - len(audio)
            audio = torch.tensor(audio, dtype=torch.float32)
            audio = F.pad(audio, (0, padding)).numpy()

        # ✅ Preprocess
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ Inference
        with torch.no_grad():
            features = wav2vec(**inputs).last_hidden_state
            output = model(features)

        probs = F.softmax(output, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

        label = "deepfake" if pred == 1 else "real"

        print(f"Prediction: {label} ({confidence:.3f})")

        return {
            "label": label,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        print("PREDICT ERROR:", repr(e))

        return {
            "label": "error",
            "confidence": 0.0
        }