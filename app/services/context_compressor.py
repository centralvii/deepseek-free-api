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
    - Следит за лимитом ~300,000 токенов (из окна в 1,000,000).
    - Гарантированно сохраняет системные инструкции и tools.
    - Гарантированно сохраняет последние N сообщений диалога с полной детализацией.
    - Сжимает / уплотняет старую середину диалога и гигантские выводы инструментов.
    """

    QWEN_MAX_WEB_TOKENS: int = 20_000
    QWEN_MAX_PAYLOAD_BYTES: int = 70_000
    DEFAULT_MAX_TOKENS: int = 300_000

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        retain_recent_count: Optional[int] = None,
        max_tool_tokens: Optional[int] = None,
    ):
        self.max_context_tokens = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 300_000)
        self.retain_recent_count = retain_recent_count or getattr(settings, "RETAIN_RECENT_MESSAGES_COUNT", 12)
        self.max_tool_tokens = max_tool_tokens or getattr(settings, "MAX_TOOL_OUTPUT_TOKENS", 25_000)

    def get_limit_for_provider(self, provider_id: str) -> int:
        """Возвращает безопасный лимит токенов контекста для конкретного провайдера."""
        if str(provider_id).lower().strip() == "qwen":
            return self.QWEN_MAX_WEB_TOKENS
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

        logger.info(
            f"Контекст диалога ({total_tokens:,} токенов) превысил порог {limit:,}. "
            f"Запуск интеллектуального сжатия..."
        )

        # 3. Разделяем на системные, старую середину и свежие сообщения
        system_msgs = [m for m in sanitized_messages if m.role == "system"]
        non_system_msgs = [m for m in sanitized_messages if m.role != "system"]

        if len(non_system_msgs) <= self.retain_recent_count:
            # Слишком мало сообщений для разделения, просто возвращаем
            return sanitized_messages

        recent_msgs = non_system_msgs[-self.retain_recent_count:]
        middle_msgs = non_system_msgs[:-self.retain_recent_count]

        # 4. Формируем сжатую сводку старой середины диалога
        summary_lines = []
        for m in middle_msgs:
            role = m.role
            c = str(m.content or "")
            if len(c) > 300:
                c = c[:280] + "..."
            summary_lines.append(f"- [{role}]: {c}")

        summary_text = (
            f"[Сводка предыдущего контекста диалога ({len(middle_msgs)} ранних сообщений сжато для оптимизации памяти)]:\n"
            + "\n".join(summary_lines)
        )

        summary_msg = OpenAIChatMessage(
            role="system",
            content=summary_text,
        )

        result = system_msgs + [summary_msg] + recent_msgs
        new_tokens = sum(estimate_tokens(m.content or "") for m in result)
        logger.info(f"✓ Контекст успешно сжат: с {total_tokens:,} до {new_tokens:,} токенов.")
        return result

    def compress_raw_prompt(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> str:
        """Сжимает текстовый промпт по токенам и байтам UTF-8 для безопасного прохождения веб-WAF."""
        if not prompt:
            return prompt

        limit = max_tokens or self.max_context_tokens
        curr_tokens = estimate_tokens(prompt)
        prompt_bytes = len(prompt.encode("utf-8"))
        effective_max_bytes = max_bytes or (self.QWEN_MAX_PAYLOAD_BYTES if limit <= self.QWEN_MAX_WEB_TOKENS else None)

        if curr_tokens <= limit and (not effective_max_bytes or prompt_bytes <= effective_max_bytes):
            return prompt

        logger.info(
            f"Промпт ({curr_tokens:,} токенов, {prompt_bytes:,} байт) превысил лимит "
            f"({limit:,} ток., {effective_max_bytes or 'unlimited'} байт). Применяется адаптивное сжатие..."
        )

        # Вычисляем целевой размер в символах с учетом байтовой плотности кодировки (UTF-8)
        bytes_per_char = max(1.0, prompt_bytes / max(1, len(prompt)))
        if effective_max_bytes and prompt_bytes > effective_max_bytes:
            target_char_len = int((effective_max_bytes - 600) / bytes_per_char)
        else:
            target_char_len = int(limit * 3.0 / bytes_per_char)

        head_chars = int(target_char_len * 0.35)
        tail_chars = int(target_char_len * 0.55)

        if head_chars + tail_chars >= len(prompt):
            return prompt

        head = prompt[:head_chars]
        tail = prompt[-tail_chars:]
        omitted_chars = len(prompt) - (head_chars + tail_chars)
        omitted_tokens = int(omitted_chars / 3.2)

        compressed = (
            f"{head}\n\n"
            f"[... ⚡ Интеллектуальное сжатие контекста: сжато {omitted_chars:,} символов (~{omitted_tokens:,} токенов) "
            f"промежуточных логов и истории для удержания фокуса модели в пределах {limit:,} токенов ...]\n\n"
            f"{tail}"
        )
        return compressed


context_compressor = ContextCompressor()
