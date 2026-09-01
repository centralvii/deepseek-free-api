from typing import Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from app.core.credentials import credentials_manager

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class TokenSetRequest(BaseModel):
    token: str = Field(..., min_length=10, description="Bearer токен из браузера (chat.deepseek.com)")


@router.post("/token", summary="Сохранить токен авторизации DeepSeek")
async def set_auth_token(request: TokenSetRequest) -> Dict[str, Any]:
    try:
        credentials_manager.save(request.token)
        return {
            "status": "success",
            "message": "Токен успешно сохранен и активирован",
            "authenticated": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить токен: {str(e)}"
        )


@router.get("/status", summary="Проверить статус авторизации")
async def get_auth_status() -> Dict[str, Any]:
    is_auth = credentials_manager.is_authenticated()
    masked_token = None
    if is_auth and credentials_manager.token:
        t = credentials_manager.token
        masked_token = t[:6] + "..." + t[-4:] if len(t) > 10 else "***"

    return {
        "authenticated": is_auth,
        "token_preview": masked_token,
    }
