from typing import Annotated, Dict, Any
from fastapi import APIRouter, Depends
import httpx
from app.api.deps import get_http_client
from app.services.session_manager import session_manager

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("/new", summary="Создать новую чат-сессию (начать диалог заново)")
async def create_new_session(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> Dict[str, Any]:
    session_id = await session_manager.create_new_session(client)
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Новая сессия успешно создана",
    }


@router.get("/current", summary="Получить ID текущей активной сессии")
async def get_current_session() -> Dict[str, Any]:
    current_id = session_manager.get_current_session_id()
    parent_id = session_manager.get_parent_message_id(current_id) if current_id else None
    return {
        "session_id": current_id,
        "parent_message_id": parent_id,
        "has_active_session": bool(current_id),
    }


@router.post("/reset", summary="Сбросить локальный контекст сессии")
async def reset_session() -> Dict[str, Any]:
    session_manager.reset_context()
    return {
        "status": "success",
        "message": "Текущий контекст сброшен. Следующий запрос создаст новую сессию.",
    }
