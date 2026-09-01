from fastapi import APIRouter
from app.api.v1.endpoints import anthropic, auth, chat, models, sessions

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(anthropic.router)
api_router.include_router(sessions.router)
api_router.include_router(models.router)
api_router.include_router(auth.router)
