from gradio_client import Client, file

client = Client("https://suramuahaha-audio-deepfake-detection.hf.space")


def predict(file_path: str):
    try:
        print("🎧 Sending to HF:", file_path)

        result = client.predict(
            file(file_path),
            api_name="/predict"
        )

        print("HF RAW RESULT:", result)  # 🔥 DEBUG

        # ✅ SAFE PARSING
        label = "real"
        confidence = 0.0

        if isinstance(result, str) and "(" in result:
            try:
                label_part, conf_part = result.strip().split("(")
                label = label_part.strip().lower()
                confidence = float(conf_part.replace(")", "").strip())
            except:
                print("⚠️ Parsing failed, using fallback")

        # fallback if parsing fails
        if label not in ["real", "deepfake"]:
            label = "real"

        confidence = max(0.0, min(1.0, float(confidence)))

        print(f"Prediction: {label} ({confidence:.3f})")

        return {
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        print("🔥 HF ERROR:", str(e))

        return {
            "label": "error",
            "confidence": 0.0
        }