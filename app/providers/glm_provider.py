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

GLM_MODELS = [
    ModelInfo(
        id="glm-5.3",
        name="GLM 5.3",
        description="Новейшая флагманская модель GLM 5.3 с мощными способностями рассуждения и генерации.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-5-pro",
        name="GLM 5 Pro",
        description="Профессиональная модель GLM-5 для сложных задач анализа и синтеза информации.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-5-coder",
        name="GLM 5 Coder",
        description="Специализированная модель линейки GLM-5 для программирования, тестирования и архитектуры ПО.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="glm-5-flash",
        name="GLM 5 Flash",
        description="Высокоскоростная модель с минимальной задержкой ответа (Zero-Latency).",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-4-plus",
        name="GLM 4 Plus",
        description="Классическая проверенная модель поколения GLM-4.",
        model_type="expert",
        supports_thinking=False,
        supports_search=True,
    ),
]


class GLMProvider(BaseLLMProvider):
    """Провайдер для GLM (chatglm.cn / BigModel API)."""

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="glm", display_name="GLM (Zhipu AI)", http_client=http_client)
        self.base_url = "https://chatglm.cn/backend-api/assistant"

    def get_models(self) -> List[ModelInfo]:
        return GLM_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("glm")

    def _resolve_glm_model(self, requested_model: str) -> str:
        req_lower = requested_model.lower().strip()
        if req_lower in ["glm-5.3", "5.3", "glm5"]:
            return "glm-5.3"
        if req_lower in ["glm-5-coder", "glm-coder", "coder"]:
            return "glm-5-coder"
        if req_lower in ["glm-5-pro", "pro"]:
            return "glm-5-pro"
        if req_lower in ["glm-5-flash", "flash"]:
            return "glm-5-flash"
        return requested_model

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        token = credentials_manager.get_token("glm")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные GLM не настроены. Укажите токен через /api/v1/auth/token?provider=glm или команду /token glm <токен>."
            )

        resolved_model = self._resolve_glm_model(request.model)
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

        url = f"{self.base_url}/stream"
        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=120.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"GLM error ({resp.status_code}): {body.decode('utf-8', errors='replace')}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"GLM API error ({resp.status_code}): {body.decode('utf-8', errors='replace')}"
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
            logger.error(f"Ошибка при вызове GLM: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка соединения с GLM API: {str(e)}"
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
            session_id=request.chat_session_id or "glm-session",
            message_id=1,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
