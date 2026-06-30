import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    APIKeyHeader,
)
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from pydantic import BaseModel

from app.database import (
    get_user_by_username,
    get_user_by_id,
    get_api_key,
    update_last_used,
)

SECRET_KEY = os.getenv("JWT_SECRET", "datadefenders-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)



class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    id: int
    username: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(username: str, user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": username,
        "user_id": user_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        if not username or user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if not user or user["id"] != user_id:
        raise credentials_exception

    return user
# API KEY

def generate_api_key() -> str:
    return "ak_live_" + secrets.token_urlsafe(32)


async def verify_api_key(
    x_api_key: str = Header(None),
):

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key missing",
        )

    key = get_api_key(x_api_key)

    if not key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    update_last_used(x_api_key)

    user = get_user_by_id(key["user_id"])

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    return user


# JWT OR API KEY

async def authenticate_request(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: str | None = Depends(api_key_header),
):
    # ---------- JWT ----------
    if credentials:
        try:
            return get_current_user(credentials)
        except HTTPException:
            pass

    # ---------- API KEY ----------
    if x_api_key:
        key = get_api_key(x_api_key)

        if key:
            update_last_used(x_api_key)

            user = get_user_by_id(key["user_id"])

            if user:
                return user

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide either Bearer token or X-API-Key.",
    )