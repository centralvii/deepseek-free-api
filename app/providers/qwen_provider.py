import json
import logging
from typing import AsyncGenerator, List, Optional
import httpx
from fastapi import HTTPException, status

from app.core.credentials import credentials_manager
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)

logger = logging.getLogger(__name__)

QWEN_MODELS = [
    ModelInfo(
        id="qwen-3.8",
        name="Qwen 3.8",
        description="Флагманская модель Qwen 3-го поколения с глубоким пониманием и рассуждениями.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3.8-coder",
        name="Qwen 3.8 Coder",
        description="Передовая специализированная модель для сложного программирования, рефакторинга и агентных пайплайнов.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="qwen-3-max",
        name="Qwen 3 Max",
        description="Максимальная по интеллектуальной мощности модель линейки Qwen 3.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-plus",
        name="Qwen 3 Plus",
        description="Сбалансированная и быстрая модель Qwen 3 общего назначения.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-flash",
        name="Qwen 3 Flash",
        description="Сверхбыстрая легковесная модель для мгновенных ответов.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-2.5-coder-32b",
        name="Qwen 2.5 Coder 32B",
        description="Классическая кодовая модель Qwen 2.5 Coder.",
        model_type="expert",
        supports_thinking=False,
        supports_search=False,
    ),
]


class QwenProvider(BaseLLMProvider):
    """Провайдер для Qwen (chat.qwen.ai / DashScope API)."""

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="qwen", display_name="Qwen (Alibaba)", http_client=http_client)
        self.base_url = "https://chat.qwen.ai/api"

    def get_models(self) -> List[ModelInfo]:
        return QWEN_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("qwen")

    def _resolve_qwen_model(self, requested_model: str) -> str:
        req_lower = requested_model.lower().strip()
        if req_lower in ["qwen-3.8-coder", "3.8-coder", "qwen-coder", "coder"]:
            return "qwen-3.8-coder"
        if req_lower in ["qwen-3.8", "3.8", "qwen3"]:
            return "qwen-3.8"
        if req_lower in ["qwen-3-max", "max", "qwen-max"]:
            return "qwen-3-max"
        if req_lower in ["qwen-3-flash", "flash", "qwen-flash"]:
            return "qwen-3-flash"
        return requested_model

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        token = credentials_manager.get_token("qwen")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные Qwen не настроены. Укажите токен через /api/v1/auth/token?provider=qwen или команду /token qwen <токен>."
            )

        resolved_model = self._resolve_qwen_model(request.model)
        headers = {
            "accept": "text/event-stream",
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=120.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"Qwen error ({resp.status_code}): {body.decode('utf-8', errors='replace')}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Qwen API error ({resp.status_code}): {body.decode('utf-8', errors='replace')}"
                )

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield StreamChunk(type="status", text="FINISHED")
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield StreamChunk(type="thinking", text=reasoning)
                            content = delta.get("content")
                            if content:
                                yield StreamChunk(type="content", text=content)
                    except Exception:
                        pass

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при вызове Qwen: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка соединения с Qwen API: {str(e)}"
            )

    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        full_thinking = []
        full_content = []
        token_usage = None

        async for chunk in self.stream_chat(request):
            if chunk.type == "thinking":
                full_thinking.append(chunk.text)
            elif chunk.type == "content":
                full_content.append(chunk.text)
            if chunk.token_usage:
                token_usage = chunk.token_usage

        return DeepSeekChatResponse(
            session_id=request.chat_session_id or "qwen-session",
            message_id=1,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
