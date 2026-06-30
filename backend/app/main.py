from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.extension import _rate_limit_exceeded_handler
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from app.api_keys import router as api_key_router
from app.auth import authenticate_request
import glob
import shutil
import os
import uuid
import subprocess
import asyncio
from contextlib import asynccontextmanager

from gradio_client import Client, handle_file
from app.database import (
    init_db,
    create_user,
    get_user_by_username,
    user_count,
    log_spam_detection,
    get_spam_history,
    is_spam_label,
    create_api_key,
    list_api_keys,
    revoke_api_key,
)
from app.auth import (
    UserCreate,
    UserLogin,
    UserPublic,
    Token,
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    generate_api_key,
    verify_api_key,
    authenticate_request,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "temp")

client = Client("suramuahaha/audio-deepfake-detection")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def seed_default_admin():
    if user_count() == 0:
        create_user("admin", hash_password("admin123"))
        print("Default admin created (username: admin, password: admin123)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_default_admin()
    yield


app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

app.add_middleware(SlowAPIMiddleware)

app.include_router(api_key_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=UserPublic)
def register(user: UserCreate):
    if len(user.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if get_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")

    user_id = create_user(user.username.strip(), hash_password(user.password))
    return UserPublic(id=user_id, username=user.username.strip())


@app.post("/auth/login", response_model=Token)
def login(credentials: UserLogin):
    user = get_user_by_username(credentials.username)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user["username"], user["id"])
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserPublic)
def me(current_user: dict = Depends(get_current_user)):
    return UserPublic(id=current_user["id"], username=current_user["username"])


@app.get("/history/spam")
def spam_history(current_user: dict = Depends(get_current_user)):
    items = get_spam_history()
    return {
        "total": len(items),
        "items": items,
    }


def resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    winget_root = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "WinGet",
        "Packages",
    )
    matches = glob.glob(
        os.path.join(winget_root, "Gyan.FFmpeg_*", "ffmpeg-*", "bin", "ffmpeg.exe")
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "ffmpeg not found. Install it with: winget install Gyan.FFmpeg"
    )


def convert_to_wav(input_path: str, output_path: str):
    ffmpeg = resolve_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        print("FFMPEG ERROR:\n", result.stderr.decode())
        raise Exception("FFmpeg conversion failed")


@app.post("/predict")
@limiter.limit("10/minute")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(authenticate_request),
):
    file_id = str(uuid.uuid4())
    filename = file.filename or "audio.webm"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in {".webm", ".wav", ".mp3"}:
        raise HTTPException(status_code=400, detail="Invalid file format")

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext == ".wav":
            wav_path = input_path
        else:
            convert_to_wav(input_path, wav_path)

        result = await asyncio.to_thread(
            client.predict,
            handle_file(wav_path),
            api_name="/predict",
        )

        label = result.get("label")
        confidence = float(result.get("confidence", 0))

        if label and is_spam_label(label):
            try:
                log_spam_detection(
                    label,
                    confidence,
                )
            except Exception as e:
                print("History save failed:", e)

        return {
            "success": True,
            "result": {
                "label": label,
                "confidence": confidence,
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "success": False,
            "error": str(e),
        }

    finally:
        for path in {input_path, wav_path}:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
