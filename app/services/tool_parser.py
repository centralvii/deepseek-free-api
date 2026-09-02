import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolCall, OpenAIToolCallFunction

logger = logging.getLogger(__name__)


def compact_tool_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Компактизирует JSON Schema инструмента для экономии токенов и лимита WAF.
    - Удаляет шумные метаданные (title, $comment, verbose descriptions)
    - Сокращает длинные описания параметров до 120 символов
    - Сохраняет критические ключи (type, properties, required, items, enum, const)
    """
    if not isinstance(schema, dict):
        return schema

    compact = {}
    for k, v in schema.items():
        if k in ("title", "$comment", "$schema", "default"):
            continue
        if k == "description" and isinstance(v, str) and len(v) > 120:
            compact[k] = v[:117] + "..."
        elif k == "properties" and isinstance(v, dict):
            compact[k] = {
                prop_name: compact_tool_schema(prop_def)
                for prop_name, prop_def in v.items()
            }
        elif k == "items" and isinstance(v, dict):
            compact[k] = compact_tool_schema(v)
        else:
            compact[k] = v
    return compact


def build_tool_system_prompt(tools: List[OpenAITool]) -> str:
    """Формирует системную инструкцию с описанием доступных инструментов."""
    raw_schema_chars = sum(
        len(json.dumps(tool.function.parameters or {}))
        for tool in tools
        if tool.type == "function" and tool.function
    )
    should_compact = raw_schema_chars > 12_000 or len(tools) > 15

    tool_lines = []
    for tool in tools:
        if tool.type == "function" and tool.function:
            fn_name = tool.function.name
            desc = str(tool.function.description or "").strip().replace("\r\n", " ")
            if len(desc) > 300:
                desc = desc[:297] + "..."
            params = tool.function.parameters or {"type": "object", "properties": {}}
            if should_compact:
                params = compact_tool_schema(params)
            params_str = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            tool_lines.append(f"## {fn_name}\nDescription: {desc}\nParameters: {params_str}")

    tools_text = "\n\n".join(tool_lines)

    prompt = f"""
# Available Tools
You have access to the following functions/tools to assist the user:

{tools_text}

# Tool Call Instructions
CRITICAL RULES FOR TOOL CALLS:
1. You ONLY REASON and REQUEST tool executions. You do NOT execute any commands or files yourself.
2. DO NOT STOP with just a text promise or declaration of intent (such as "Изучу файлы...", "I will check...", "Let me read..."). When you need to inspect, read, search, edit, or run something, you MUST output the tool call in the SAME response!
3. NEVER simulate, guess, or fabricate command or tool output — output the tool call and wait for the actual result from the system.
4. When requesting a tool, output valid JSON inside `<tool_call>...</tool_call>`:
<tool_call>
{{"name": "<function_name>", "arguments": {{...}}}}
</tool_call>

Alternatively, standard JSON format is also accepted:
{{"tool_call": {{"name": "<function_name>", "arguments": {{...}}}}}}

5. If no tool call is needed and the entire task is complete, provide your normal conversational response directly.
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

    # 3. Если последнее сообщение — вывод инструмента (tool output),
    # добавляем директиву немедленно продолжить и вызвать инструмент, а не останавливаться на обещании
    if compressed_messages and compressed_messages[-1].role in ["tool", "function"]:
        prompt_parts.append(
            "\n[System Directive: The previous tool execution has finished and its output is provided above. Proceed with the task immediately. If you need to inspect more files or run commands, invoke the tool call NOW: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. Do NOT stop with only a conversational promise or intent.]"
        )

    full_prompt = "\n\n".join(prompt_parts)
    if max_tokens:
        full_prompt = context_compressor.compress_raw_prompt(full_prompt, max_tokens=max_tokens)
    return full_prompt


def normalize_qwen_parameter_tags(text: str) -> str:
    """Нормализует гибридные теги параметров Qwen (<parameter=key>...</parameter>) в валидный JSON."""
    if not text or "parameter" not in text:
        return text

    pattern = r'<parameter=([a-zA-Z0-9_]+)>\s*([\s\S]*?)\s*</parameter>'

    def _replace_param(match):
        param_name = match.group(1)
        val = match.group(2).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith('{') and val.endswith('}')) or (val.startswith('[') and val.endswith(']')):
            return f'"{param_name}": {val},'
        elif val.lower() in ("true", "false", "null") or val.isdigit():
            return f'"{param_name}": {val},'
        else:
            safe_val = json.dumps(val, ensure_ascii=False)
            return f'"{param_name}": {safe_val},'

    normalized = re.sub(pattern, _replace_param, text)
    return normalized


def repair_json_argument_strings(raw_json_str: str) -> str:
    """Исправляет распространенные ошибки экранирования кавычек внутри полей аргументов."""
    match = re.search(r'("command"|"content"|"text")\s*:\s*"(.*)"\s*,\s*"description"', raw_json_str, re.DOTALL)
    if match:
        field_name = match.group(1)
        inner_content = match.group(2)
        escaped_content = inner_content.replace('\\"', '"').replace('"', '\\"')
        repaired = raw_json_str[:match.start(2)] + escaped_content + raw_json_str[match.end(2):]
        return repaired
    return raw_json_str


def parse_tool_call_json(json_str: str) -> Optional[Tuple[str, str]]:
    """Пытается распарсить JSON строку вызова инструмента."""
    json_str = json_str.strip()
    if not json_str:
        return None

    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        json_str_norm = normalize_qwen_parameter_tags(json_str)
        try:
            data = json.loads(json_str_norm)
        except json.JSONDecodeError:
            repaired_str = repair_json_argument_strings(json_str_norm)
            try:
                data = json.loads(repaired_str)
            except json.JSONDecodeError:
                pass

    if isinstance(data, dict):
        if "tool_call" in data and isinstance(data["tool_call"], dict):
            data = data["tool_call"]

        if "name" in data and ("arguments" in data or "parameters" in data):
            fn_name = data.get("name")
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = "{}"
            return str(fn_name), args_str

    return None


def extract_tool_calls(text: str) -> Tuple[str, List[OpenAIToolCall]]:
    """
    Универсальный парсер вызовов инструментов для DeepSeek и Qwen.
    Поддерживает форматы:
    - <tool_call> ... </tool_call> (включая непарные кавычки <tool_call">)
    - Несколько JSON объектов подряд внутри одного <tool_call>
    - DeepSeek DSML: <｜DSML｜tool_calls> ... <｜DSML｜invoke name="..."> ...
    - Claude/DeepSeek XML: <invoke name="..."> <parameter name="..."> ...
    - Стандартный OpenAI тег: <function=name> ... </function>
    - Голый JSON без тегов: {"name": "...", "arguments": {...}}
    """
    if not text:
        return text, []

    tool_calls: List[OpenAIToolCall] = []
    seen_calls = set()

    def add_call(name: str, args_str: str) -> bool:
        name = name.strip()
        if not name:
            return False
        key = (name, args_str.strip())
        if key in seen_calls:
            return False
        seen_calls.add(key)
        call_id = f"call_{len(tool_calls)+1}_{abs(hash(key)) % 1000000}"
        tool_calls.append(
            OpenAIToolCall(
                id=call_id,
                type="function",
                function=OpenAIToolCallFunction(
                    name=name,
                    arguments=args_str,
                )
            )
        )
        return True

    clean_text = text

    # --- 1. DeepSeek DSML Markup (<｜DSML｜tool_calls> ... <｜DSML｜invoke name="...">) ---
    dsml_pattern = r'<[｜|]DSML[｜|]tool_calls>([\s\S]*?)<[｜|]DSML[｜|]tool_calls>'
    dsml_matches = list(re.finditer(dsml_pattern, clean_text))
    if dsml_matches:
        for m in dsml_matches:
            block = m.group(1)
            invokes = re.finditer(r'<[｜|]DSML[｜|]invoke name="([^"]+)">([\s\S]*?)</[｜|]DSML[｜|]invoke>', block)
            for inv in invokes:
                fn_name = inv.group(1)
                inv_body = inv.group(2)
                params_dict = {}
                param_matches = re.finditer(r'<[｜|]DSML[｜|]parameter name="([^"]+)"[^>]*>([\s\S]*?)</[｜|]DSML[｜|]parameter>', inv_body)
                for pm in param_matches:
                    p_name = pm.group(1)
                    p_val = pm.group(2).strip()
                    try:
                        p_parsed = json.loads(p_val)
                        params_dict[p_name] = p_parsed
                    except Exception:
                        params_dict[p_name] = p_val
                add_call(fn_name, json.dumps(params_dict, ensure_ascii=False))
            clean_text = clean_text.replace(m.group(0), "")

    # --- 2. Claude/DeepSeek XML Invoke (<invoke name="..."> <parameter name="..."> ...) ---
    xml_invoke_pattern = r'<invoke name="([^"]+)">([\s\S]*?)</invoke>'
    xml_invokes = list(re.finditer(xml_invoke_pattern, clean_text))
    if xml_invokes:
        for inv in xml_invokes:
            fn_name = inv.group(1)
            inv_body = inv.group(2)
            params_dict = {}
            param_matches = re.finditer(r'<parameter name="([^"]+)">([\s\S]*?)</parameter>', inv_body)
            for pm in param_matches:
                p_name = pm.group(1)
                p_val = pm.group(2).strip()
                try:
                    p_parsed = json.loads(p_val)
                    params_dict[p_name] = p_parsed
                except Exception:
                    params_dict[p_name] = p_val
            add_call(fn_name, json.dumps(params_dict, ensure_ascii=False))
            clean_text = clean_text.replace(inv.group(0), "")

    # Очищаем обрамляющие теги XML вызовов
    clean_text = re.sub(r'</?(?:tool_call|tool_calls|tools)>', '', clean_text)

    # --- 3. <tool_call> ... </tool_call> (с поддержкой нескольких JSON внутри) ---
    tool_call_regex = r'<tool_call["\']?>([\s\S]*?)</tool_call["\']?>'
    tool_matches = list(re.finditer(tool_call_regex, clean_text, re.IGNORECASE))
    if tool_matches:
        for match in tool_matches:
            content = match.group(1).strip()
            parsed_any = False

            res = parse_tool_call_json(content)
            if res:
                fn_name, args_str = res
                if add_call(fn_name, args_str):
                    parsed_any = True

            if not parsed_any:
                json_candidates = re.finditer(r'\{[\s\S]*?\}', content)
                for cand in json_candidates:
                    c_str = cand.group(0)
                    res_c = parse_tool_call_json(c_str)
                    if res_c:
                        fn_name, args_str = res_c
                        add_call(fn_name, args_str)

            clean_text = clean_text.replace(match.group(0), "")

    # --- 4. <function=name> ... </function> ---
    fn_tag_regex = r'<function=([a-zA-Z0-9_]+)>([\s\S]*?)</function>'
    fn_matches = list(re.finditer(fn_tag_regex, clean_text, re.IGNORECASE))
    if fn_matches:
        for match in fn_matches:
            fn_name = match.group(1)
            content = match.group(2).strip()
            try:
                norm_content = normalize_qwen_parameter_tags(content)
                args_dict = json.loads(norm_content)
                args_str = json.dumps(args_dict, ensure_ascii=False)
            except Exception:
                args_str = content
            add_call(fn_name, args_str)
            clean_text = clean_text.replace(match.group(0), "")

    # --- 5. Голый JSON без тегов: {"name": "...", "arguments": {...}} ---
    if not tool_calls:
        naked_json_regex = r'\{\s*"name"\s*:\s*"([a-zA-Z0-9_]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}'
        naked_matches = list(re.finditer(naked_json_regex, clean_text))
        if naked_matches:
            for nm in naked_matches:
                fn_name = nm.group(1)
                args_str = nm.group(2)
                try:
                    args_dict = json.loads(args_str)
                    args_str = json.dumps(args_dict, ensure_ascii=False)
                except Exception:
                    pass
                add_call(fn_name, args_str)
                clean_text = clean_text.replace(nm.group(0), "")

    # --- 6. Голый JSON с неэкранированными кавычками внутри "command": "..." ---
    if not tool_calls:
        loose_regex = r'\{\s*"name"\s*:\s*"([a-zA-Z0-9_]+)"\s*,\s*"arguments"\s*:\s*\{([\s\S]*?)\}\s*\}'
        loose_matches = list(re.finditer(loose_regex, clean_text))
        for lm in loose_matches:
            fn_name = lm.group(1)
            raw_full = lm.group(0)
            repaired = repair_json_argument_strings(raw_full)
            res = parse_tool_call_json(repaired)
            if res:
                add_call(res[0], res[1])
                clean_text = clean_text.replace(raw_full, "")

    clean_text = clean_text.strip()
    return clean_text, tool_calls
