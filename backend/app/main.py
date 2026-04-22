from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import subprocess
import asyncio
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

# ✅ Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ✅ Health check
@app.get("/")
def health():
    return {"status": "ok"}


# ✅ Convert audio to WAV
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


# ✅ Prediction endpoint
@app.post("/predict")
async def detect(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        # ✅ Validate file
        if not file.filename.endswith((".webm", ".wav", ".mp3")):
            raise HTTPException(status_code=400, detail="Invalid file format")

        # ✅ Save uploaded file
        with open(webm_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Convert to WAV
        convert_to_wav(webm_path, wav_path)

        # ✅ Call HF model (non-blocking)
        result = await asyncio.to_thread(predict, wav_path)

        return {
            "success": True,
            "label": result.get("label"),
            "confidence": float(result.get("confidence", 0))
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # ✅ Cleanup temp files
        for path in [webm_path, wav_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass