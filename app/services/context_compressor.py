import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.core.config import settings
from app.schemas.openai import OpenAIChatMessage

logger = logging.getLogger(__name__)


def estimate_tokens(text: Union[str, Any]) -> int:
    """
    Быстрая и точная оценка количества токенов для многоязычного текста, кода и JSON.
    Учитывает:
    - Английский текст и код: ~3.8 символа на токен
    - Кириллица / Русский: ~1.8 символа на токен
    - Китайские иероглифы (CJK): ~1.2 символа на токен
    - Пробелы и спецсимволы
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)

    length = len(text)
    if length == 0:
        return 0

    # Подсчет символов не-ASCII (кириллица, CJK)
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    ascii_count = length - non_ascii_count

    # Оценка: ASCII ~ 3.8 символов/токен, Non-ASCII ~ 1.8 символов/токен
    tokens = int((ascii_count / 3.8) + (non_ascii_count / 1.8))
    return max(1, tokens)


def truncate_tool_output(content: str, max_tokens: int = 25_000) -> str:
    """
    Сжимает гигантские выводы инструментов (дампы файлов, большие логи),
    сохраняя начало (Head) и конец (Tail) вывода.
    """
    current_tokens = estimate_tokens(content)
    if current_tokens <= max_tokens:
        return content

    # Оставляем 40% сверху и 40% снизу, вырезаем середину
    target_char_len = int(max_tokens * 3.0)
    head_len = int(target_char_len * 0.45)
    tail_len = int(target_char_len * 0.45)

    head = content[:head_len]
    tail = content[-tail_len:]
    omitted_chars = len(content) - (head_len + tail_len)
    omitted_tokens = int(omitted_chars / 3.0)

    return (
        f"{head}\n\n"
        f"[... ⚠️ Контекстный компрессор: пропущено {omitted_chars:,} символов (~{omitted_tokens:,} токенов) середины вывода ...]\n\n"
        f"{tail}"
    )


class ContextCompressor:
    """
    Интеллектуальный менеджер сжатия контекста:
    - Следит за лимитом токенов провайдеров.
    - Гарантированно сохраняет системные инструкции и tools.
    - Гарантированно сохраняет последние N сообщений диалога с полной детализацией.
    - Сжимает / уплотняет старую середину диалога и гигантские выводы инструментов.
    """

    QWEN_MAX_WEB_TOKENS: int = 20_000
    QWEN_MAX_PAYLOAD_BYTES: int = 70_000
    DEEPSEEK_MAX_WEB_TOKENS: int = 50_000
    DEFAULT_MAX_TOKENS: int = 50_000

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        retain_recent_count: Optional[int] = None,
        max_tool_tokens: Optional[int] = None,
    ):
        self.max_context_tokens = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 50_000)
        self.retain_recent_count = retain_recent_count or getattr(settings, "RETAIN_RECENT_MESSAGES_COUNT", 12)
        self.max_tool_tokens = max_tool_tokens or getattr(settings, "MAX_TOOL_OUTPUT_TOKENS", 25_000)

    def get_limit_for_provider(self, provider_id: str) -> int:
        """Возвращает безопасный лимит токенов контекста для конкретного провайдера."""
        pid = str(provider_id).lower().strip()
        if pid == "qwen":
            return self.QWEN_MAX_WEB_TOKENS
        elif pid == "deepseek":
            return self.DEEPSEEK_MAX_WEB_TOKENS
        return self.max_context_tokens

    def compress_openai_messages(
        self,
        messages: List[OpenAIChatMessage],
        max_tokens: Optional[int] = None,
    ) -> List[OpenAIChatMessage]:
        """Сжимает список сообщений OpenAI до допустимого бюджета токенов."""
        if not messages:
            return messages

        limit = max_tokens or self.max_context_tokens
        tool_limit = min(self.max_tool_tokens, max(2_000, limit // 5))
        
        # 1. Сначала сжимаем гигантские tool выводы
        sanitized_messages: List[OpenAIChatMessage] = []
        for msg in messages:
            if msg.role in ["tool", "function"] and isinstance(msg.content, str):
                compressed_content = truncate_tool_output(msg.content, max_tokens=tool_limit)
                if compressed_content != msg.content:
                    # Создаем копию с усеченным контентом
                    msg_dict = msg.model_dump()
                    msg_dict["content"] = compressed_content
                    sanitized_messages.append(OpenAIChatMessage(**msg_dict))
                    continue
            sanitized_messages.append(msg)

        # 2. Оцениваем общий объем
        total_tokens = sum(estimate_tokens(m.content or "") for m in sanitized_messages)
        if total_tokens <= limit:
            return sanitized_messages

        # 3. Разделяем на: Системные + Старая история + Свежие сообщения (неприкосновенные)
        system_msgs = [m for m in sanitized_messages if m.role == "system"]
        non_system = [m for m in sanitized_messages if m.role != "system"]

        recent_count = min(self.retain_recent_count, len(non_system))
        middle_msgs = non_system[:-recent_count] if recent_count > 0 else []
        recent_msgs = non_system[-recent_count:] if recent_count > 0 else non_system

        # 4. Если всё еще превышает лимит, агрессивно сжимаем старые сообщения из middle
        budget_for_middle = limit - sum(estimate_tokens(m.content or "") for m in system_msgs + recent_msgs)
        if budget_for_middle <= 0:
            # Бюджет исчерпан свежими сообщениями, оставляем только их + системные
            return system_msgs + recent_msgs

        compressed_middle: List[OpenAIChatMessage] = []
        middle_tokens = 0
        # Идем с конца middle_msgs (наиболее свежие из старых)
        for msg in reversed(middle_msgs):
            t = estimate_tokens(msg.content or "")
            if middle_tokens + t <= budget_for_middle:
                compressed_middle.append(msg)
                middle_tokens += t
            else:
                # Если сообщение слишком велико, сжимаем его
                remaining = budget_for_middle - middle_tokens
                if remaining > 200 and isinstance(msg.content, str):
                    shortened = truncate_tool_output(msg.content, max_tokens=remaining)
                    msg_dict = msg.model_dump()
                    msg_dict["content"] = shortened
                    compressed_middle.append(OpenAIChatMessage(**msg_dict))
                break

        compressed_middle.reverse()
        return system_msgs + compressed_middle + recent_msgs

    def compress_raw_prompt(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Сжимает сырую строку промпта, если она превышает заданный лимит.
        ВАЖНО: Заголовок промпта (# Available Tools, # Tool Call Instructions, System Instructions)
        НЕЛЬЗЯ обрезать, иначе модель теряет правила вызова инструментов.
        Сжатие применяется строго к блоку 'Conversation History:'.
        """
        limit = max_tokens or self.max_context_tokens
        current_tokens = estimate_tokens(prompt)
        if current_tokens <= limit:
            return prompt

        history_marker = "\n\nConversation History:\n"
        if history_marker in prompt:
            header, history = prompt.split(history_marker, 1)
            header_tokens = estimate_tokens(header)
            available_for_history = max(1_000, limit - header_tokens)

            # Если история превышает доступный лимит, сжимаем ее середину
            history_tokens = estimate_tokens(history)
            if history_tokens > available_for_history:
                # Оставляем 20% начала истории и 70% самого конца (самые актуальные шаги)
                ratio = available_for_history / history_tokens
                keep_head_chars = int(len(history) * ratio * 0.20)
                keep_tail_chars = int(len(history) * ratio * 0.70)

                cut_history = (
                    f"{history[:keep_head_chars]}\n\n"
                    f"[... ⚠️ Контекстный компрессор: пропущена старая история диалога ...]\n\n"
                    f"{history[-keep_tail_chars:]}"
                )
                return f"{header}{history_marker}{cut_history}"

        # Резервный срез (сохраняя начало с описанием инструментов)
        ratio = limit / current_tokens
        keep_head_chars = int(len(prompt) * ratio * 0.40)
        keep_tail_chars = int(len(prompt) * ratio * 0.50)
        return (
            f"{prompt[:keep_head_chars]}\n\n"
            f"[... ⚠️ Контекстный компрессор: пропущено из-за лимита токенов ...]\n\n"
            f"{prompt[-keep_tail_chars:]}"
        )


context_compressor = ContextCompressor()
