from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from app.core.credentials import credentials_manager

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class TokenSetRequest(BaseModel):
    token: str = Field(..., min_length=5, description="Bearer токен из браузера или API")
    provider: Optional[str] = Field(default="deepseek", description="Провайдер: deepseek, qwen, glm")


@router.post("/token", summary="Сохранить токен авторизации для указанного провайдера")
async def set_auth_token(request: TokenSetRequest) -> Dict[str, Any]:
    prov = (request.provider or "deepseek").lower().strip()
    try:
        credentials_manager.save(request.token, provider=prov)
        return {
            "status": "success",
            "provider": prov,
            "message": f"Токен для провайдера '{prov}' успешно сохранен и активирован",
            "authenticated": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить токен для {prov}: {str(e)}"
        )


@router.get("/status", summary="Проверить статус авторизации всех провайдеров")
async def get_auth_status() -> Dict[str, Any]:
    tokens = credentials_manager.load()
    providers_status = {}

    for p in ["deepseek", "qwen", "glm"]:
        t = tokens.get(p)
        is_auth = bool(t)
        masked = (t[:6] + "..." + t[-4:]) if (is_auth and len(t) > 10) else ("***" if is_auth else None)
        providers_status[p] = {
            "authenticated": is_auth,
            "token_preview": masked,
        }

    return {
        "authenticated": any(ps["authenticated"] for ps in providers_status.values()),
        "providers": providers_status,
    }
