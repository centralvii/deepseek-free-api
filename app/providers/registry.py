import logging
from typing import Dict, List, Optional
import httpx
from app.providers.base import BaseLLMProvider
from app.providers.deepseek_provider import DeepSeekProvider
from app.providers.qwen_provider import QwenProvider
from app.schemas.chat import ModelInfo

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Центральный реестр и диспетчер LLM-провайдеров."""

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider_id: str = "deepseek"

    def init_providers(self, http_client: httpx.AsyncClient) -> None:
        """Инициализирует доступных провайдеров с общим HTTP клиентом."""
        self._providers["deepseek"] = DeepSeekProvider(http_client)
        self._providers["qwen"] = QwenProvider(http_client)

    def get_provider(self, provider_id: Optional[str] = None) -> BaseLLMProvider:
        """Возвращает провайдера по ID или провайдера по умолчанию."""
        pid = (provider_id or self.default_provider_id).lower().strip()
        if pid not in self._providers:
            raise KeyError(f"Неизвестный провайдер '{pid}'. Доступные: {list(self._providers.keys())}")
        return self._providers[pid]

    def set_default_provider(self, provider_id: str) -> None:
        pid = provider_id.lower().strip()
        if pid not in self._providers:
            raise ValueError(f"Неизвестный провайдер '{pid}'. Доступные: {list(self._providers.keys())}")
        self.default_provider_id = pid
        logger.info(f"Активный провайдер по умолчанию изменен на: {pid}")

    def resolve_provider_for_model(self, model_name: str) -> BaseLLMProvider:
        """
        Автоматически определяет нужного провайдера по имени модели:
        - qwen-... -> Qwen
        - deepseek-..., r1, chat, search, etc. -> DeepSeek (или default_provider)
        """
        m_lower = model_name.lower().strip()

        if m_lower.startswith("qwen") or "qwen-" in m_lower or m_lower in ["3.8", "qwen3"]:
            return self._providers["qwen"]

        if m_lower.startswith("deepseek") or m_lower in ["r1", "reasoner", "expert", "v4-pro", "v4-flash", "v4-vision"]:
            return self._providers["deepseek"]

        # Если не совпало ни с одним префиксом, используем текущего провайдера по умолчанию
        return self.get_provider(self.default_provider_id)

    def get_all_models(self) -> List[ModelInfo]:
        """Возвращает общий объединенный список моделей всех провайдеров."""
        all_models: List[ModelInfo] = []
        for prov in self._providers.values():
            all_models.extend(prov.get_models())
        return all_models

    def list_providers(self) -> List[Dict[str, str]]:
        return [
            {
                "id": p.provider_id,
                "name": p.display_name,
                "is_default": p.provider_id == self.default_provider_id,
                "authenticated": p.is_authenticated(),
            }
            for p in self._providers.values()
        ]


provider_registry = ProviderRegistry()
