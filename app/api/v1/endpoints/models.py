from typing import Annotated, Dict, Any, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_http_client
from app.providers.registry import provider_registry
from app.schemas.chat import ModelInfo
import httpx

router = APIRouter(tags=["Models & Providers"])


class ProviderSwitchRequest(BaseModel):
    provider: str


@router.get("/api/v1/models", response_model=List[ModelInfo], summary="Список доступных моделей всех провайдеров")
async def list_models(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> List[ModelInfo]:
    return provider_registry.get_all_models()


@router.get("/v1/models", summary="OpenAI-совместимый список моделей всех провайдеров")
async def openai_list_models(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> Dict[str, Any]:
    models = provider_registry.get_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": 1700000000,
                "owned_by": m.id.split("-")[0] if "-" in m.id else "llm",
                "permission": [],
                "root": m.id,
                "parent": None,
            }
            for m in models
        ],
    }


@router.get("/api/v1/providers", summary="Список доступных LLM-провайдеров и их статус")
async def list_providers(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> Dict[str, Any]:
    return {
        "default_provider": provider_registry.default_provider_id,
        "providers": provider_registry.list_providers(),
    }


@router.post("/api/v1/providers/switch", summary="Переключить активного провайдера по умолчанию")
async def switch_provider(
    request: ProviderSwitchRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> Dict[str, Any]:
    try:
        provider_registry.set_default_provider(request.provider)
        return {
            "status": "success",
            "default_provider": provider_registry.default_provider_id,
            "message": f"Провайдер успешно изменен на {provider_registry.default_provider_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
