from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user, generate_api_key
from app.database import (
    create_api_key,
    list_api_keys,
    revoke_api_key,
)

router = APIRouter(prefix="/api-key", tags=["API Keys"])


class APIKeyCreate(BaseModel):
    name: str = "Default Key"


@router.post("/create")
def create_key(
    payload: APIKeyCreate,
    current_user: dict = Depends(get_current_user),
):
    key = generate_api_key()

    create_api_key(
        user_id=current_user["id"],
        api_key=key,
        name=payload.name,
    )

    return {
        "message": "API key created successfully",
        "api_key": key,
    }


@router.get("/list")
def get_keys(
    current_user: dict = Depends(get_current_user),
):
    return list_api_keys(current_user["id"])


@router.delete("/{key_id}")
def delete_key(
    key_id: int,
    current_user: dict = Depends(get_current_user),
):
    keys = list_api_keys(current_user["id"])

    if not any(k["id"] == key_id for k in keys):
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    revoke_api_key(key_id)

    return {
        "message": "API key revoked successfully"
    }