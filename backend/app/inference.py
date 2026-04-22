from gradio_client import Client, file

# 🔥 Initialize once (global client)
client = Client("https://suramuahaha-audio-deepfake-detection.hf.space")


def predict(file_path: str):
    try:
        print("🎧 Sending to HF:", file_path)

        # 🔥 Call Hugging Face Space
        result = client.predict(
            file(file_path),
            api_name="/predict"
        )

        # result format: "real (0.91)"
        label, confidence = result.split(" (")
        confidence = float(confidence[:-1])

        print(f"Prediction: {label} ({confidence:.3f})")

        return {
            "label": label,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        print("🔥 HF ERROR:", str(e))

        return {
            "label": "error",
            "confidence": 0.0
        }