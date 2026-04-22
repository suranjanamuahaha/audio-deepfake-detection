from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import subprocess
from app.inference import predict

app = FastAPI()

# ✅ CORS (open for hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Stable paths (Railway-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ✅ Health check
@app.get("/")
def health():
    return {"status": "ok"}


# ✅ Convert to WAV using ffmpeg
def convert_to_wav(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print("❌ FFMPEG ERROR:\n", result.stderr.decode())
        raise Exception("FFmpeg conversion failed")


@app.post("/predict")
async def detect(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        # ✅ Save uploaded file
        with open(webm_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not os.path.exists(webm_path):
            raise Exception("WEBM file not saved")

        # ✅ Convert to WAV
        convert_to_wav(webm_path, wav_path)

        if not os.path.exists(wav_path):
            raise Exception("WAV conversion failed")

        # ✅ Run model
        result = predict(wav_path)

        return {
            "success": True,
            "label": result.get("label"),
            "confidence": float(result.get("confidence", 0))
        }

    except Exception as e:
        print("🔥 ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ Always cleanup
        for path in [webm_path, wav_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass