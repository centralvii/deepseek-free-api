from typing import Annotated, Dict, Any, List
from fastapi import APIRouter, Depends
from app.api.deps import get_deepseek_client
from app.schemas.chat import ModelInfo
from app.services.deepseek_client import DeepSeekClient

router = APIRouter(tags=["Models"])


@router.get("/api/v1/models", response_model=List[ModelInfo], summary="Список доступных моделей DeepSeek")
async def list_models(
    client: Annotated[DeepSeekClient, Depends(get_deepseek_client)]
) -> List[ModelInfo]:
    return client.get_models()


@router.get("/v1/models", summary="OpenAI-совместимый список моделей")
async def openai_list_models(
    client: Annotated[DeepSeekClient, Depends(get_deepseek_client)]
) -> Dict[str, Any]:
    models = client.get_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
                "permission": [],
                "root": m.id,
                "parent": None,
            }
            for m in models
        ],
    }
