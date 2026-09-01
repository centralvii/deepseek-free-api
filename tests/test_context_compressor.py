import pytest
from app.schemas.openai import OpenAIChatMessage
from app.services.context_compressor import ContextCompressor, estimate_tokens, truncate_tool_output


def test_estimate_tokens():
    # ASCII text
    ascii_text = "Hello world! This is a test."
    tokens_ascii = estimate_tokens(ascii_text)
    assert tokens_ascii > 0

    # Cyrillic text
    cyrillic_text = "Привет мир! Это тестовая строка на русском языке."
    tokens_cyr = estimate_tokens(cyrillic_text)
    assert tokens_cyr > 0
    assert tokens_cyr > len(cyrillic_text) / 2.5


def test_truncate_tool_output():
    short_output = "Short output"
    assert truncate_tool_output(short_output, max_tokens=100) == short_output

    huge_output = "A" * 50_000
    truncated = truncate_tool_output(huge_output, max_tokens=500)
    assert len(truncated) < len(huge_output)
    assert "Контекстный компрессор" in truncated


def test_compress_openai_messages():
    compressor = ContextCompressor(max_context_tokens=300, retain_recent_count=2)

    system_msg = OpenAIChatMessage(role="system", content="You are a helpful coding assistant.")
    
    # Создаем длинную историю сообщений
    messages = [system_msg]
    for i in range(20):
        messages.append(OpenAIChatMessage(role="user", content=f"User message number {i} " + "content " * 20))
        messages.append(OpenAIChatMessage(role="assistant", content=f"Assistant response number {i} " + "answer " * 20))

    compressed = compressor.compress_openai_messages(messages, max_tokens=300)

    # 1. Системный промпт сохранен
    assert compressed[0].role == "system"
    assert compressed[0].content == system_msg.content

    # 2. Появилась сводка
    assert any("Сводка предыдущего контекста диалога" in (m.content or "") for m in compressed)

    # 3. Последние 2 сообщения сохранены без изменений
    assert compressed[-1].content == messages[-1].content
    assert compressed[-2].content == messages[-2].content


def test_compress_raw_prompt():
    compressor = ContextCompressor(max_context_tokens=100)
    short_prompt = "Short prompt"
    assert compressor.compress_raw_prompt(short_prompt, max_tokens=100) == short_prompt

    huge_prompt = "Header instructions:\n" + ("Data line details\n" * 500) + "\nFinal task: summarize everything."
    compressed = compressor.compress_raw_prompt(huge_prompt, max_tokens=50)

    assert "Интеллектуальное сжатие контекста" in compressed
    assert "Header instructions" in compressed
    assert "Final task" in compressed
