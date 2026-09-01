import logging
from typing import Dict, Optional
import httpx
from app.core.config import settings
from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self._current_session_id: Optional[str] = None
        self._last_message_ids: Dict[str, Optional[int]] = {}
        self._session_titles: Dict[str, str] = {}

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def set_current_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id
        if session_id not in self._last_message_ids:
            self._last_message_ids[session_id] = None

    def get_parent_message_id(self, session_id: str) -> Optional[int]:
        return self._last_message_ids.get(session_id)

    def update_session_state(self, session_id: str, last_message_id: int, title: Optional[str] = None) -> None:
        self._last_message_ids[session_id] = last_message_id
        if title:
            self._session_titles[session_id] = title

    async def create_new_session(self, client: httpx.AsyncClient) -> str:
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
        self._last_message_ids[session_id] = None
        logger.info(f"Создана новая сессия DeepSeek: {session_id}")
        return session_id

    async def get_or_create_session(self, client: httpx.AsyncClient, session_id: Optional[str] = None) -> str:
        if session_id:
            if session_id not in self._last_message_ids:
                self._last_message_ids[session_id] = None
            return session_id

        if self._current_session_id:
            return self._current_session_id

        return await self.create_new_session(client)

    def reset_context(self) -> None:
        self._current_session_id = None


session_manager = SessionManager()
