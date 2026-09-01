from typing import AsyncGenerator, List
import httpx
from app.core.credentials import credentials_manager
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)
from app.services.deepseek_client import AVAILABLE_MODELS, DeepSeekClient


class DeepSeekProvider(BaseLLMProvider):
    """Провайдер для DeepSeek Web API (chat.deepseek.com)."""

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="deepseek", display_name="DeepSeek", http_client=http_client)
        self.client_impl = DeepSeekClient(http_client)

    def get_models(self) -> List[ModelInfo]:
        return AVAILABLE_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("deepseek")

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        async for chunk in self.client_impl.stream_chat(request):
            yield chunk

    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        return await self.client_impl.send_message(request)
