import datetime
import json
import logging
from typing import AsyncGenerator, List, Optional
import uuid
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
    """
    Провайдер для прямого веб-API chat.qwen.ai (/api/v2/chat/completions).
    Полностью воспроизводит заголовки, куки и формат запросов реального веб-клиента.
    """

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="qwen", display_name="Qwen (Alibaba)", http_client=http_client)
        self.base_url = "https://chat.qwen.ai"

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

    def _build_headers(self, token_or_cookie: str, chat_id: str, thinking_enabled: bool) -> dict:
        """Формирует заголовки браузера для chat.qwen.ai."""
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%a %b %d %Y %H:%M:%S GMT+0000")
        req_id = str(uuid.uuid4())

        # Если передан полный заголовок Cookie
        if "token=" in token_or_cookie or "; " in token_or_cookie or "_bl_uid=" in token_or_cookie:
            cookie_header = token_or_cookie
            jwt_token = ""
            for part in token_or_cookie.split(";"):
                part = part.strip()
                if part.startswith("token="):
                    jwt_token = part[6:].strip()
        else:
            jwt_token = token_or_cookie.strip()
            think_mode = "Thinking" if thinking_enabled else "Normal"
            cookie_header = f"token={jwt_token}; qwen-thinking_mode={think_mode}; qwen-locale=ru-RU; qwen-theme=dark;"

        headers = {
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Cookie": cookie_header,
            "Host": "chat.qwen.ai",
            "Origin": "https://chat.qwen.ai",
            "Referer": f"https://chat.qwen.ai/c/{chat_id}",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Timezone": now_str,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
            "Version": "0.2.89",
            "X-Accel-Buffering": "no",
            "X-Request-Id": req_id,
            "bx-v": "2.5.37",
            "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "source": "web",
        }

        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"

        return headers

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        token = credentials_manager.get_token("qwen")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные Qwen не настроены. Укажите токен или Cookie через команду /token qwen <токен_или_куки>."
            )

        chat_id = request.chat_session_id or str(uuid.uuid4())
        resolved_model = self._resolve_qwen_model(request.model)
        thinking_enabled = request.thinking_enabled if request.thinking_enabled is not None else True

        headers = self._build_headers(token, chat_id, thinking_enabled)

        # Тело запроса к Qwen v2
        payload = {
            "stream": True,
            "incremental": True,
            "model": resolved_model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                    "chat_type": "t2t",
                    "feature_config": {
                        "thinking_enabled": thinking_enabled,
                    }
                }
            ],
            "parent_id": None,
        }

        url = f"{self.base_url}/api/v2/chat/completions?chat_id={chat_id}"

        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=120.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                err_text = body.decode("utf-8", errors="replace")
                logger.error(f"Qwen error ({resp.status_code}): {err_text}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Qwen API error ({resp.status_code}): {err_text}"
                )

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield StreamChunk(type="status", text="FINISHED", session_id=chat_id)
                        break
                    try:
                        data = json.loads(data_str)
                        # 1. Формат choices (OpenAI/v2 style)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Блок рассуждений (thinking / reasoning_content / thought)
                            reasoning = delta.get("reasoning_content") or delta.get("thought") or delta.get("thinking")
                            if reasoning:
                                yield StreamChunk(type="thinking", text=reasoning, session_id=chat_id)
                            # Основной контент ответа
                            content = delta.get("content")
                            if content:
                                yield StreamChunk(type="content", text=content, session_id=chat_id)

                        # 2. Формат response / output (Qwen native v2)
                        elif "response" in data and isinstance(data["response"], dict):
                            resp_obj = data["response"]
                            if "thinking" in resp_obj and resp_obj["thinking"]:
                                yield StreamChunk(type="thinking", text=resp_obj["thinking"], session_id=chat_id)
                            if "content" in resp_obj and resp_obj["content"]:
                                yield StreamChunk(type="content", text=resp_obj["content"], session_id=chat_id)

                        elif "output" in data and isinstance(data["output"], dict):
                            out_text = data["output"].get("text", "")
                            if out_text:
                                yield StreamChunk(type="content", text=out_text, session_id=chat_id)
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
        session_id = request.chat_session_id or str(uuid.uuid4())

        async for chunk in self.stream_chat(request):
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.type == "thinking":
                full_thinking.append(chunk.text)
            elif chunk.type == "content":
                full_content.append(chunk.text)
            if chunk.token_usage:
                token_usage = chunk.token_usage

        return DeepSeekChatResponse(
            session_id=session_id,
            message_id=1,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
