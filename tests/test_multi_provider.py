import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry


@pytest.mark.asyncio
async def test_provider_registry_routing():
    providers = provider_registry.list_providers()
    provider_ids = [p["id"] for p in providers]
    assert "deepseek" in provider_ids
    assert "qwen" in provider_ids
    assert "glm" not in provider_ids

    p_deepseek = provider_registry.resolve_provider_for_model("deepseek-v4-pro")
    assert p_deepseek.provider_id == "deepseek"

    p_qwen = provider_registry.resolve_provider_for_model("qwen-3.8-coder")
    assert p_qwen.provider_id == "qwen"


@pytest.mark.asyncio
async def test_all_models_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        model_ids = [m["id"] for m in models]

        assert "deepseek-v4-pro" in model_ids
        assert "deepseek-v4-flash" in model_ids
        assert "deepseek-reasoner" in model_ids

        assert "qwen-3.8" in model_ids
        assert "qwen-3.8-coder" in model_ids
        assert "qwen3.7-plus" in model_ids


@pytest.mark.asyncio
async def test_provider_switching_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_provider" in data
        assert len(data["providers"]) == 2

        sw_resp = await ac.post("/api/v1/providers/switch", json={"provider": "qwen"})
        assert sw_resp.status_code == 200
        assert sw_resp.json()["default_provider"] == "qwen"
        assert provider_registry.default_provider_id == "qwen"

        sw_resp2 = await ac.post("/api/v1/providers/switch", json={"provider": "deepseek"})
        assert sw_resp2.status_code == 200
        assert sw_resp2.json()["default_provider"] == "deepseek"


@pytest.mark.asyncio
async def test_multi_provider_credentials():
    orig_qwen = credentials_manager.get_token("qwen")

    try:
        credentials_manager._tokens["qwen"] = "mock_qwen_token"

        assert credentials_manager.get_token("qwen") == "mock_qwen_token"
        assert credentials_manager.is_authenticated("qwen") is True
    finally:
        if orig_qwen:
            credentials_manager._tokens["qwen"] = orig_qwen


@pytest.mark.asyncio
async def test_qwen_adaptive_context_compression():
    """Проверяет, что контекст для Qwen автоматически укладывается в безопасный лимит WAF."""
    from app.services.context_compressor import context_compressor, estimate_tokens
    from app.services.tool_parser import format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    # Создаем 50 инструментов
    tools = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name=f"tool_{i}",
                description=f"Description of tool {i} for testing context size compression",
                parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            )
        )
        for i in range(50)
    ]

    # Создаем длинную историю сообщений (>30,000 токенов)
    messages = [
        OpenAIChatMessage(role="system", content="Ты системный помощник."),
        OpenAIChatMessage(role="user", content="Инструкция: " + ("Очень длинный текст задачи агента " * 3000)),
    ]

    qwen_limit = context_compressor.get_limit_for_provider("qwen")
    assert qwen_limit == 20_000

    compiled = format_messages_to_prompt(messages, tools, max_tokens=qwen_limit)
    compiled_tokens = estimate_tokens(compiled)

    # Проверяем, что промпт уложился в безопасный лимит
    assert compiled_tokens <= qwen_limit * 1.5
    # И размер в байтах безопасен для WAF (<80 KB)
    assert len(compiled.encode("utf-8")) < 80_000


@pytest.mark.asyncio
async def test_stream_error_sse_formatting(monkeypatch):
    """Проверяет, что ошибка провайдера безопасно передается в SSE без падения ASGI."""
    from fastapi import HTTPException

    qwen_p = provider_registry.get_provider("qwen")

    async def mock_fail(*args, **kwargs):
        raise HTTPException(status_code=403, detail="WAF challenge error")
        yield

    monkeypatch.setattr(qwen_p, "stream_chat", mock_fail)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        req_payload = {
            "model": "qwen-3.8-coder",
            "messages": [{"role": "user", "content": "Привет"}],
            "stream": True,
        }
        resp = await ac.post("/v1/chat/completions", json=req_payload)
        assert resp.status_code == 200
        text = resp.text
        assert "WAF challenge error" in text
        assert "data: [DONE]" in text


def test_robust_tool_call_extraction():
    """Тестирует парсинг вызовов инструментов с многострочным кодом, опечатками в тегах и дедупликацией."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """
Вот созданный файл:

<tool_call">
{"name": "write_to_file", "arguments": {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"}}
</tool_call">

<tool_call>
{"name": "write_to_file", "arguments": {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"}}
</tool_call>

<function=replace_in_file>
{"path": "main.py", "diff": "- old\n+ new"}
</function>
"""
    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 2  # Дубликат write_to_file отсеян, добавлен replace_in_file
    assert calls[0].function.name == "write_to_file"
    assert "calculator.py" in calls[0].function.arguments
    assert calls[1].function.name == "replace_in_file"
    assert "main.py" in calls[1].function.arguments
    assert "<tool_call" not in clean_text
    assert "<function=" not in clean_text


def test_multiple_json_in_single_tool_call_tag():
    """Тестирует парсинг нескольких JSON объектов внутри одного тега <tool_call> (как возвращает Qwen)."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Изучу структуру проекта и текущий парсер курсов.

<tool_call>
{"name": "Bash", "arguments": {"command": "find \\"E:/vibecoding/stepik-searcher\\" -type f | head -80", "description": "List project files"}}
{"name": "Bash", "arguments": {"command": "ls -la \\"E:/vibecoding/stepik-searcher\\"", "description": "List root directory"}}
</tool_call>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 2
    assert calls[0].function.name == "Bash"
    assert "find" in calls[0].function.arguments
    assert calls[1].function.name == "Bash"
    assert "ls -la" in calls[1].function.arguments
    assert clean_text == "Изучу структуру проекта и текущий парсер курсов."
    assert "<tool_call>" not in clean_text
