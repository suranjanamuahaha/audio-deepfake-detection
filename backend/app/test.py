import requests

url = "https://audio-deepfake-detection-uco.onrender.com/predict"

with open("test.mp3", "rb") as f:
    res = requests.post(url, files={"file": f})

print(res.status_code)
print(res.text)