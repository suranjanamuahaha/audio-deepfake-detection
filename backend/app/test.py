from gradio_client import Client, file

client = Client("https://suramuahaha-audio-deepfake-detection.hf.space")

result = client.predict(
    file("test.mp3"),   # 🔥 THIS IS THE FIX
    api_name="/predict"
)

print(result)