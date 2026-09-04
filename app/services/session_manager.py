import logging
from typing import Dict, Optional
import httpx
from app.core.config import settings
from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


class SessionManager:
    """Управляет жизненным циклом чат-сессий и отслеживает контекст (parent_message_id).
    
    В режиме single_session_mode сессия хранится ОТДЕЛЬНО для каждого провайдера.
    При смене провайдера (deepseek → qwen → deepseek) восстанавливается прежняя сессия.
    """

    def __init__(self):
        # Старый одиночный ID (DeepSeek legacy) — теперь хранится в _provider_sessions["deepseek"]
        self._current_session_id: Optional[str] = None
        # Сессии per-provider: provider_id -> session_id
        self._provider_sessions: Dict[str, Optional[str]] = {}
        # Сопоставление session_id -> last_message_id (для сохранения контекста диалога)
        self._last_message_ids: Dict[str, Optional[int]] = {}
        # Заголовки чатов
        self._session_titles: Dict[str, str] = {}
        # Режим единой сессии (single) vs изолированные чаты (multi)
        self.single_session_mode: bool = bool(
            settings.SINGLE_SESSION_MODE or (settings.PROXY_MODE.lower() == "single")
        )

    def set_single_session_mode(self, enabled: bool) -> None:
        """Включает или выключает режим единой сессии (без создания новых чатов)."""
        self.single_session_mode = enabled
        logger.info(f"Режим сессий переключен: {'Единая сессия (Single)' if enabled else 'Изолированные чаты (Multi)'}")

    def is_single_session_mode(self) -> bool:
        return self.single_session_mode

    # ── Per-provider session storage ─────────────────────────────────────────

    def get_provider_session(self, provider_id: str) -> Optional[str]:
        """Возвращает сохранённый session_id для конкретного провайдера."""
        sid = self._provider_sessions.get(provider_id)
        # Обратная совместимость: DeepSeek раньше использовал _current_session_id
        if sid is None and provider_id == "deepseek":
            sid = self._current_session_id
        return sid

    def set_provider_session(self, provider_id: str, session_id: str) -> None:
        """Сохраняет session_id для конкретного провайдера."""
        self._provider_sessions[provider_id] = session_id
        if provider_id == "deepseek":
            self._current_session_id = session_id
        if session_id not in self._last_message_ids:
            self._last_message_ids[session_id] = None
        logger.debug(f"Сохранена сессия провайдера {provider_id}: {session_id}")

    def clear_provider_session(self, provider_id: str) -> None:
        """Сбрасывает session_id конкретного провайдера (начать диалог заново)."""
        old = self._provider_sessions.pop(provider_id, None)
        if provider_id == "deepseek":
            self._current_session_id = None
        if old:
            logger.info(f"Сессия провайдера {provider_id} сброшена: {old}")

    # ── DeepSeek legacy API (обратная совместимость) ────────────────────────

    def invalidate_current_session(self) -> None:
        """Инвалидирует текущую сессию при ошибке или устаревании на сервере."""
        if self._current_session_id:
            logger.warning(f"Инвалидация сессии DeepSeek: {self._current_session_id}")
            self._last_message_ids.pop(self._current_session_id, None)
            self._provider_sessions.pop("deepseek", None)
            self._current_session_id = None

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def set_current_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id
        self._provider_sessions["deepseek"] = session_id
        if session_id not in self._last_message_ids:
            self._last_message_ids[session_id] = None

    def get_parent_message_id(self, session_id: str) -> Optional[int]:
        return self._last_message_ids.get(session_id)

    def update_session_state(self, session_id: str, last_message_id: int, title: Optional[str] = None) -> None:
        self._last_message_ids[session_id] = last_message_id
        if title:
            self._session_titles[session_id] = title

    async def create_new_session(self, client: httpx.AsyncClient) -> str:
        """Создает новую сессию через веб-API DeepSeek."""
        url = f"{settings.DEEPSEEK_BASE_URL}/api/v0/chat_session/create"
        headers = {
            "accept": "*/*",
            "authorization": credentials_manager.auth_header,
            "content-type": "application/json",
            "x-client-bundle-id": settings.CLIENT_BUNDLE_ID,
            "x-client-locale": settings.CLIENT_LOCALE,
            "x-client-platform": settings.CLIENT_PLATFORM,
            "x-client-timezone-offset": settings.CLIENT_TIMEZONE_OFFSET,
            "x-client-version": settings.CLIENT_VERSION,
            "user-agent": settings.USER_AGENT,
        }

        response = await client.post(url, json={}, headers=headers)
        response.raise_for_status()

        result = response.json() or {}
        data = result.get("data") if isinstance(result, dict) else {}
        if data is None:
            data = {}
        biz_data = data.get("biz_data") if isinstance(data, dict) else {}
        if biz_data is None:
            biz_data = {}

        session_id = None
        if isinstance(biz_data, dict):
            if "chat_session" in biz_data and isinstance(biz_data["chat_session"], dict):
                session_id = biz_data["chat_session"].get("id")
            elif "id" in biz_data:
                session_id = biz_data.get("id")

        if not isinstance(result, dict) or result.get("code") != 0 or not session_id:
            raise ValueError(f"Не удалось создать чат-сессию DeepSeek: {result}")

        self._current_session_id = session_id
        self._provider_sessions["deepseek"] = session_id
        self._last_message_ids[session_id] = None
        logger.info(f"Создана новая сессия DeepSeek: {session_id}")
        return session_id

    async def get_or_create_session(self, client: httpx.AsyncClient, session_id: Optional[str] = None) -> str:
        """
        Возвращает указанный session_id, либо использует текущую единую сессию (в режиме single_session_mode),
        либо создает новую чистую сессию (в режиме multi_session_mode).
        """
        if session_id:
            if session_id not in self._last_message_ids:
                self._last_message_ids[session_id] = None
            self._current_session_id = session_id
            self._provider_sessions["deepseek"] = session_id
            return session_id

        # В режиме единой сессии (single) повторно используем сохранённую сессию DeepSeek
        if self.single_session_mode:
            saved = self._provider_sessions.get("deepseek") or self._current_session_id
            if saved:
                logger.debug(f"Переиспользование текущей сессии DeepSeek (Single-Session): {saved}")
                self._current_session_id = saved
                return saved

        # Иначе создаем новую сессию
        return await self.create_new_session(client)

    def reset_context(self) -> None:
        """Сбрасывает текущую активную сессию DeepSeek (для начала чистого диалога)."""
        self._current_session_id = None
        self._provider_sessions.pop("deepseek", None)


session_manager = SessionManager()
