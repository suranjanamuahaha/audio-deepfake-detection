from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth import (
    UserCreate,
    UserLogin,
    Token,
    UserPublic,
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

from app.database import (
    init_db,
    create_user,
    get_user_by_username,
    get_spam_history,
    log_spam_detection,
    is_spam_label,
)

import shutil
import os
import uuid
import subprocess

app = FastAPI()

optional_auth = HTTPBearer(auto_error=False)

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Create Gradio client once
# -------------------------
client = Client("suramuahaha/audio-deepfake-detection")

init_db()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/auth/register", response_model=UserPublic)
def register(user: UserCreate):
    existing = get_user_by_username(user.username)

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Username already exists",
        )

    user_id = create_user(
        user.username,
        hash_password(user.password),
    )

    return UserPublic(
        id=user_id,
        username=user.username,
    )


@app.post("/auth/login", response_model=Token)
def login(user: UserLogin):
    db_user = get_user_by_username(user.username)

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    if not verify_password(
        user.password,
        db_user["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    return Token(
        access_token=create_access_token(
            db_user["username"],
            db_user["id"],
        )
    )


@app.get("/auth/me", response_model=UserPublic)
def current_user(
    user=Depends(get_current_user),
):
    return UserPublic(
        id=user["id"],
        username=user["username"],
    )


@app.get("/history/spam")
def spam_history(
    user=Depends(get_current_user),
):
    items = get_spam_history(user["id"])

    return {
        "items": items,
        "total": len(items),
    }

def convert_to_wav(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        print(result.stderr.decode())
        raise Exception("FFmpeg conversion failed")


@app.post("/predict")
async def detect(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_auth),
):
    file_id = str(uuid.uuid4())

    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        if not file.filename.endswith((".webm", ".wav", ".mp3")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format",
            )

        with open(webm_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        convert_to_wav(webm_path, wav_path)

        # Call the Hugging Face Space
        result = client.predict(
            handle_file(wav_path),
            api_name="/predict",
        )

        label = result.get("label")
        confidence = result.get("confidence", 0)

        # Save spam history only for authenticated users
        if credentials:
            try:
                user = get_current_user(credentials)

                if is_spam_label(label):
                    log_spam_detection(
                        user["id"],
                        label,
                        confidence,
                    )

            except Exception as e:
                print("SAVE ERROR:", repr(e))
                raise

        return {
            "success": True,
            "result": result,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print(e)

        return {
            "success": False,
            "error": str(e),
        }

    finally:
        for path in [webm_path, wav_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass