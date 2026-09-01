from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
import httpx
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)


class BaseLLMProvider(ABC):
    """Абстрактный базовый класс для всех LLM-провайдеров (DeepSeek, Qwen, GLM)."""

    def __init__(self, provider_id: str, display_name: str, http_client: httpx.AsyncClient):
        self.provider_id = provider_id
        self.display_name = display_name
        self.client = http_client

    @abstractmethod
    def get_models(self) -> List[ModelInfo]:
        """Возвращает список поддерживаемых моделей провайдера."""
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Проверяет наличие токена авторизации для данного провайдера."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Потоковый вызов чата (генератор StreamChunk)."""
        pass

    @abstractmethod
    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        """Синхронный вызов чата."""
        pass
