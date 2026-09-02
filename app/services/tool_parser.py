import json
import re
import uuid
from typing import List, Optional, Tuple, Any, Dict
from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolCall, OpenAIToolCallFunction


def build_tool_system_prompt(tools: List[OpenAITool]) -> str:
    """Формирует системную инструкцию с описанием доступных инструментов (JSON Schema)."""
    tools_definitions = []
    for tool in tools:
        if tool.type == "function" and tool.function:
            tools_definitions.append({
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": tool.function.parameters or {"type": "object", "properties": {}},
                }
            })

    tools_json = json.dumps(tools_definitions, ensure_ascii=False, separators=(",", ":"))

    prompt = f"""
# Available Tools
You have access to the following functions/tools to assist the user:

```json
{tools_json}
```

# Tool Call Instructions
CRITICAL REQUIREMENT:
- When the user asks you to create, write, edit, replace, or modify files, or execute commands, you MUST NOT merely output code or commands in markdown.
- You MUST invoke the appropriate tool using a `<tool_call>` block with a valid JSON object containing `"name"` and `"arguments"`.
- DIRECT TOOL EXECUTION: When performing file operations (reading, writing, editing, listing, or searching files) or terminal commands, invoke the direct tool (such as Read, Write, Edit, Bash, Glob, Grep) directly. Do NOT delegate file operations to subagent tools (like `Agent` or `Task`).
- PURE JSON FORMAT: Output clean, strictly valid JSON inside `<tool_call>`. Never output XML parameter tags (such as `<parameter=...>` or `</parameter>`).

Example format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

If no tool call is needed, provide your normal conversational response directly.
"""
    return prompt.strip()


def format_messages_to_prompt(
    messages: List[OpenAIChatMessage],
    tools: Optional[List[OpenAITool]] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Преобразует историю сообщений OpenAI (system, user, assistant, tool)
    в единый контекстный промпт для веб-интерфейса DeepSeek/Qwen.
    Автоматически применяет сжатие контекста при превышении лимита токенов.
    """
    from app.services.context_compressor import context_compressor
    compressed_messages = context_compressor.compress_openai_messages(messages, max_tokens=max_tokens)

    prompt_parts = []

    # 1. Если переданы tools, добавляем системную инструкцию по инструментам
    if tools:
        tool_instruction = build_tool_system_prompt(tools)
        prompt_parts.append(tool_instruction)

    # 2. Обрабатываем системные и пользовательские сообщения
    system_messages = []
    history_messages = []

    for msg in compressed_messages:
        role = msg.role
        content = msg.content or ""
        if isinstance(content, list):
            # Если переданы multipart сообщения (текст + изображения)
            text_pieces = []
            for piece in content:
                if isinstance(piece, dict) and piece.get("type") == "text":
                    text_pieces.append(piece.get("text", ""))
            content = " ".join(text_pieces)

        if role == "system":
            system_messages.append(content)
        elif role == "user":
            history_messages.append(f"User: {content}")
        elif role == "assistant":
            if msg.tool_calls:
                tc_str = ""
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments
                    tc_str += f"\n<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {fn_args}}}\n</tool_call>"
                history_messages.append(f"Assistant: {content}{tc_str}")
            else:
                history_messages.append(f"Assistant: {content}")
        elif role in ["tool", "function"]:
            tool_id = msg.tool_call_id or msg.name or "tool"
            history_messages.append(f"Tool [{tool_id}] Output:\n{content}")

    if system_messages:
        prompt_parts.append("System Instructions:\n" + "\n".join(system_messages))

    if history_messages:
        prompt_parts.append("\nConversation History:\n" + "\n".join(history_messages))

    full_prompt = "\n\n".join(prompt_parts)
    if max_tokens:
        full_prompt = context_compressor.compress_raw_prompt(full_prompt, max_tokens=max_tokens)
    return full_prompt


def normalize_qwen_parameter_tags(text: str) -> str:
    """Нормализует гибридные теги параметров Qwen (<parameter=key>...</parameter>) в валидный JSON."""
    if not text or "parameter" not in text:
        return text
    # 1. Замена перехода между параметрами: </parameter>\n<parameter=key> -> ", "key": 
    normalized = re.sub(r'\s*</parameter>\s*<parameter=([a-zA-Z0-9_\-]+)>\s*', r'", "\1": ', text)
    # 2. Замена одиночного </parameter> -> "
    normalized = re.sub(r'\s*</parameter>', r'"', normalized)
    # 3. Замена одиночного <parameter=key> -> "key": 
    normalized = re.sub(r'<parameter=([a-zA-Z0-9_\-]+)>\s*', r'"\1": ', normalized)
    return normalized


def _parse_all_tool_json(raw_json: str) -> List[Tuple[str, str]]:
    """Парсит все JSON-объекты (один или несколько параллельных) из блока tool_call."""
    results: List[Tuple[str, str]] = []
    if not raw_json:
        return results

    s = normalize_qwen_parameter_tags(raw_json.strip())
    decoder = json.JSONDecoder(strict=False)
    idx = 0
    while idx < len(s):
        while idx < len(s) and s[idx].isspace():
            idx += 1
        if idx >= len(s):
            break
        try:
            obj, end_idx = decoder.raw_decode(s, idx)
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments") or obj.get("parameters") or obj.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
            idx = end_idx
        except Exception:
            next_brace = s.find('{', idx + 1)
            if next_brace != -1:
                idx = next_brace
            else:
                break

    # Fallback 1: стандартный json.loads
    if not results:
        try:
            data = json.loads(s, strict=False)
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or data.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
        except Exception:
            pass

    # Fallback 2: экранируем сырые переносы строк
    if not results:
        try:
            sanitized = re.sub(r'[\r\n]+', '\\n', s)
            data = json.loads(sanitized, strict=False)
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or data.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
        except Exception:
            pass

    return results


def extract_tool_calls(text: str) -> Tuple[str, List[OpenAIToolCall]]:
    """
    Извлекает вызовы инструментов из ответа модели:
    - Поддерживает один или несколько JSON-объектов внутри одного тега <tool_call>...</tool_call>.
    - Поддерживает гибридные теги параметров Qwen (<parameter=key>...</parameter>).
    - Поддерживает теги <tool_call>...</tool_call> (включая <tool_call"> и с атрибутами).
    - Поддерживает markdown блоки ```tool_call...```.
    - Поддерживает нативный формат Qwen <function=name>...</function>.
    - Поддерживает многострочный код внутри аргументов (strict=False).
    - Выполняет дедупликацию идентичных вызовов.
    Возвращает (очищенный_текст, список_tool_calls).
    """
    tool_calls: List[OpenAIToolCall] = []
    seen_calls = set()
    clean_text = text

    # 0. Проверка формата DeepSeek DSML: <｜DSML｜tool_calls>...<｜DSML｜invoke name="...">...</｜DSML｜invoke>...</｜DSML｜tool_calls>
    dsml_invoke_pat = r"<[｜\|]*\s*DSML\s*[｜\|]*invoke\s+name=[\"']?([^\"'>]+)[\"']?[^>]*>\s*(.*?)\s*</[｜\|]*\s*DSML\s*[｜\|]*invoke>"
    dsml_param_pat = r"<[｜\|]*\s*DSML\s*[｜\|]*parameter\s+name=[\"']?([^\"'>]+)[\"']?[^>]*>\s*(.*?)\s*</[｜\|]*\s*DSML\s*[｜\|]*parameter>"

    for match in re.finditer(dsml_invoke_pat, text, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2).strip()
        args_dict = {}
        for pm in re.finditer(dsml_param_pat, body, re.DOTALL):
            p_name = pm.group(1).strip()
            p_val = pm.group(2).strip()
            if (p_val.startswith("{") and p_val.endswith("}")) or (p_val.startswith("[") and p_val.endswith("]")):
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
            args_dict[p_name] = p_val

        args_str = json.dumps(args_dict, ensure_ascii=False)
        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )

    clean_text = re.sub(r"<[｜\|]*\s*DSML\s*[｜\|]*tool_calls?>.*?</[｜\|]*\s*DSML\s*[｜\|]*tool_calls?>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(dsml_invoke_pat, "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"</?[｜\|]*\s*DSML\s*[｜\|]*[^>]*>", "", clean_text)

    # 1. Проверка формата Claude / Anthropic / DeepSeek: <invoke name="...">...</invoke>
    invoke_pat = r"<invoke\s+name=[\"']?([a-zA-Z0-9_\-\.]+)[\"']?[^>]*>\s*(.*?)\s*</invoke>"
    param_pat = r"<parameter\s+(?:name=[\"']?([a-zA-Z0-9_\-]+)[\"']?|=([a-zA-Z0-9_\-]+)|([a-zA-Z0-9_\-]+))[^>]*>\s*(.*?)\s*</parameter>"

    for match in re.finditer(invoke_pat, text, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2).strip()
        args_dict = {}
        for pm in re.finditer(param_pat, body, re.DOTALL):
            p_name = pm.group(1) or pm.group(2) or pm.group(3)
            p_val = pm.group(4).strip()
            if (p_val.startswith("{") and p_val.endswith("}")) or (p_val.startswith("[") and p_val.endswith("]")):
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
            args_dict[p_name] = p_val

        args_str = json.dumps(args_dict, ensure_ascii=False)
        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )

    # Очищаем блоки invoke (включая если они обернуты в <tool_call>...<invoke>...</tool_calls>)
    clean_text = re.sub(r"<tool_calls?[^>]*>\s*(?:<invoke\b.*?</invoke>\s*)+</tool_calls?>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(invoke_pat, "", clean_text, flags=re.DOTALL)

    # 2. Паттерны для поиска стандартных блоков JSON tool_call
    patterns = [
        r"<tool_calls?[^>]*>\s*(.*?)\s*</tool_calls?[^>]*>",
        r"```(?:tool_call|tool_calls|function_call)\s*(.*?)\s*```",
    ]

    for pat in patterns:
        for match in re.finditer(pat, text, re.DOTALL):
            raw_content = match.group(1)
            parsed_list = _parse_all_tool_json(raw_content)
            for name, args_str in parsed_list:
                call_key = (name, args_str)
                if call_key not in seen_calls:
                    seen_calls.add(call_key)
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls.append(
                        OpenAIToolCall(
                            id=call_id,
                            type="function",
                            function=OpenAIToolCallFunction(name=name, arguments=args_str),
                        )
                    )
            clean_text = re.sub(pat, "", clean_text, flags=re.DOTALL)

    # 3. Проверка нативного формата Qwen: <function=name>args</function>
    func_pat = r"<function=([a-zA-Z0-9_\-\.]+)[^>]*>\s*(.*?)\s*</function>"
    for match in re.finditer(func_pat, text, re.DOTALL):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        args_str = raw_args
        if "<parameter" in raw_args:
            param_dict = {}
            for p in re.finditer(r'<parameter=([a-zA-Z0-9_\-]+)>\s*(.*?)\s*(?:</parameter>|$)', raw_args, re.DOTALL):
                param_dict[p.group(1)] = p.group(2).strip().strip("\"'")
            if param_dict:
                args_str = json.dumps(param_dict, ensure_ascii=False)
        else:
            try:
                args_obj = json.loads(raw_args, strict=False)
                args_str = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, dict) else str(args_obj)
            except Exception:
                args_str = raw_args

        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )
    clean_text = re.sub(func_pat, "", clean_text, flags=re.DOTALL)

    return clean_text.strip(), tool_calls
