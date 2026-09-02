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


def test_hybrid_qwen_parameter_tags_extraction():
    """Тестирует парсинг вызовов инструментов, где Qwen подмешивает теги <parameter=key> и </parameter>."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """
<tool_call>
{"name": "Bash", "arguments": {"command": "find /e/vibecoding/stepik-searcher -type f -name \\"*.py\\" -o -name \\"*.js\\" -o -name \\"*.ts\\" -o -name \\"*.json\\" | head -50
</parameter>
<parameter=description> "List project files to understand structure"}}
</tool_call>
"""
    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "find /e/vibecoding" in calls[0].function.arguments
    assert "List project files to understand structure" in calls[0].function.arguments
    assert "<parameter" not in clean_text


def test_deepseek_claude_xml_invoke_extraction():
    """Тестирует парсинг вызовов инструментов в формате Claude/DeepSeek XML (<invoke name=...>)."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Let me start by exploring the project.

<tool_call>
<invoke name="Bash">
<parameter name="command">cd /e/vibecoding/stepik-searcher && git ls-files | head -200</parameter>
<parameter name="description">List tracked files in project</parameter>
</invoke>
</tool_calls>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "cd /e/vibecoding/stepik-searcher" in calls[0].function.arguments
    assert "List tracked files in project" in calls[0].function.arguments
    assert clean_text == "Let me start by exploring the project."
    assert "<invoke" not in clean_text
    assert "<tool_call" not in clean_text
    assert "<tool_calls" not in clean_text


def test_deepseek_dsml_tool_calls_extraction():
    """Тестирует парсинг вызовов инструментов в формате DeepSeek Markup Language (DSML)."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Let me explore the project.
<｜DSML｜tool_calls>
    <｜DSML｜invoke name="Bash">
        <｜DSML｜parameter name="command" string="true">cd /e/vibecoding/stepik-searcher && git status</｜DSML｜parameter>
        <｜DSML｜parameter name="description" string="true">Check git status</｜DSML｜parameter>
    </｜DSML｜invoke>
</｜DSML｜tool_calls>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "cd /e/vibecoding/stepik-searcher" in calls[0].function.arguments
    assert "Check git status" in calls[0].function.arguments
    assert clean_text == "Let me explore the project."
    assert "DSML" not in clean_text
    assert "<｜" not in clean_text


def test_naked_json_tool_call_with_unescaped_quotes():
    """Тестирует извлечение голого JSON без тегов tool_call и с неэкранированными кавычками внутри команды."""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Изучу проект, чтобы понять текущую структуру парсера.

{"name": "Bash", "arguments": {"command":"cd /e/vibecoding/stepik-searcher && ls -la && echo "---TRACKED---" && git ls-files | grep -v '^"' | grep -vi 'FILES' | head -200","description":"List project files excluding noisy FILES dir"}}"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "cd /e/vibecoding/stepik-searcher" in calls[0].function.arguments
    assert "List project files" in calls[0].function.arguments
    assert clean_text == "Изучу проект, чтобы понять текущую структуру парсера."
    assert "{" not in clean_text
    assert "Bash" not in clean_text


def test_compact_tool_schema():
    """Тестирует компактное сжатие JSON Schema инструментов."""
    from app.services.tool_parser import compact_tool_schema

    schema = {
        "type": "object",
        "title": "ToolArguments",
        "description": "Top-level description of tool arguments",
        "properties": {
            "command": {
                "type": "string",
                "title": "CommandTitle",
                "description": "A very long detailed description of what this command is going to do when executed on the local terminal environment in milliseconds.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
            },
        },
        "required": ["command"],
    }

    compacted = compact_tool_schema(schema)
    assert compacted["type"] == "object"
    assert "properties" in compacted
    assert compacted["properties"]["command"]["type"] == "string"
    assert "title" not in compacted["properties"]["command"]
    assert compacted["properties"]["command"]["description"].endswith("...")
    assert len(compacted["properties"]["command"]["description"]) <= 120
    assert compacted["required"] == ["command"]


def test_system_directive_after_tool_output():
    """Тестирует добавление системной директивы при завершении вывода инструмента."""
    from app.services.tool_parser import format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    messages = [
        OpenAIChatMessage(role="user", content="Найди файлы проекта"),
        OpenAIChatMessage(role="assistant", content="Запускаю поиск"),
        OpenAIChatMessage(role="tool", tool_call_id="call_1", content="file1.py\nfile2.py"),
    ]
    tools = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name="Bash",
                description="Run shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        )
    ]

    prompt = format_messages_to_prompt(messages, tools)
    assert "[System Directive:" in prompt
    assert "Do NOT stop with only a conversational promise" in prompt


def test_intent_pattern_matching():
    """Тестирует определение обещаний действия для Continuation Recovery."""
    from app.api.v1.endpoints.chat import INTENT_PAT

    sample1 = "Изучил структуру проекта. Теперь мне нужно понять, как в Stepik API представлены задания со стоимостью и текст заданий. Изучу оставшиеся файлы бэкенда и фронтенд-структуру."
    sample2 = "Let me check the backend code to understand how endpoints are configured."
    sample3 = "Вот готовый результат работы программы. Всего хорошего!"
    sample4 = "Let me study the remaining backend files and frontend structure to fully understand the project before planning."

    assert INTENT_PAT.search(sample1) is not None
    assert INTENT_PAT.search(sample2) is not None
    assert INTENT_PAT.search(sample3) is None
    assert INTENT_PAT.search(sample4) is not None


def test_raw_file_call_recovery():
    """Тестирует авто-извлечение и конвертацию вызовов Edit/Write из неформатированного текста <tool_call> path code."""
    import json
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Now I have a complete picture. Let me improve the parser.

<tool_call>

E:\\vibecoding\\stepik-searcher\\backend\\app\\services\\stepik.py
# 4. Fetch steps (tasks) in chunks
tasks = []
for i in range(0, len(step_ids), 100):
    cost = s.get("worth", 0)

# 4. Fetch steps (tasks) in chunks
tasks = []
for i in range(0, len(step_ids), 100):
    cost = s.get("cost", s.get("worth", 0))
    text_plain = _html_to_plain_text(text)
"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Edit"
    args = json.loads(calls[0].function.arguments)
    assert "stepik.py" in args["file_path"]
    assert "s.get(\"worth\", 0)" in args["old_string"]
    assert "s.get(\"cost\"" in args["new_string"]
    assert "Now I have a complete picture" in clean_text
    assert "<tool_call>" not in clean_text
