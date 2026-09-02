import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTool,
    AnthropicUsage,
)
from app.schemas.chat import DeepSeekChatRequest, DeepSeekChatResponse
from app.services.tool_parser import compact_tool_schema, extract_tool_calls


def build_anthropic_tools_prompt(tools: List[AnthropicTool]) -> str:
    """Генерирует системную инструкцию инструментов из формата Anthropic."""
    raw_schema_chars = sum(
        len(json.dumps(tool.input_schema or {}))
        for tool in tools
    )
    should_compact = raw_schema_chars > 12_000 or len(tools) > 15

    tools_definitions = []
    for tool in tools:
        desc = str(tool.description or "").strip()
        if len(desc) > 350:
            desc = desc[:347] + "..."
        params = tool.input_schema or {"type": "object", "properties": {}}
        if should_compact:
            params = compact_tool_schema(params)

        tools_definitions.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": desc,
                "parameters": params,
            }
        })

    tools_json = json.dumps(tools_definitions, ensure_ascii=False, separators=(",", ":"))

    return f"""
# Available Tools
You have access to the following functions/tools to assist the user:

```json
{tools_json}
```

# Tool Call Instructions
CRITICAL RULES:
1. You ONLY REASON and REQUEST tool executions. You do NOT execute any commands or files yourself.
2. When performing actions (reading, writing, editing files, searching, running terminal commands), REQUEST the tool call.
3. NEVER simulate, guess, or fabricate command or tool output — output the tool call and wait for the actual result from the system.
4. When requesting a tool, output ONLY the tool call. Do NOT add explanation, text, or markdown before or after the tool call.
5. PURE JSON FORMAT: Output strictly valid JSON inside `<tool_call>...</tool_call>`:
<tool_call>
{{"name": "<function_name>", "arguments": {{...}}}}
</tool_call>

Alternatively, standard JSON format is also accepted:
{{"tool_call": {{"name": "<function_name>", "arguments": {{...}}}}}}

If no tool call is needed, provide your normal conversational response directly.
""".strip()


def convert_anthropic_request_to_deepseek(request: AnthropicMessagesRequest) -> Tuple[DeepSeekChatRequest, bool]:
    """
    Конвертирует запрос Anthropic Messages API в DeepSeekChatRequest:
    - Извлекает system prompt
    - Распаковывает блоки content (text, image, tool_use, tool_result)
    - Добавляет описания tools в нативном формате DeepSeek
    """
    prompt_parts = []

    # 1. Инструкция по инструментам
    has_tools = bool(request.tools)
    if request.tools:
        prompt_parts.append(build_anthropic_tools_prompt(request.tools))

    # 2. Системный промпт (в Anthropic передается отдельно)
    if request.system:
        if isinstance(request.system, str):
            prompt_parts.append(f"System Instructions:\n{request.system.strip()}")
        elif isinstance(request.system, list):
            sys_texts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    sys_texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    sys_texts.append(block)
            if sys_texts:
                prompt_parts.append("System Instructions:\n" + "\n".join(sys_texts))

    # 3. Сообщения диалога
    history_messages = []
    for msg in request.messages:
        role = msg.role
        role_label = "User" if role == "user" else "Assistant"

        if isinstance(msg.content, str):
            history_messages.append(f"{role_label}: {msg.content}")
        elif isinstance(msg.content, list):
            block_texts = []
            for block in msg.content:
                if isinstance(block, str):
                    block_texts.append(block)
                elif isinstance(block, dict):
                    b_type = block.get("type", "text")
                    if b_type == "text":
                        block_texts.append(block.get("text", ""))
                    elif b_type == "thinking":
                        block_texts.append(f"[Thinking: {block.get('thinking', '')}]")
                    elif b_type == "tool_use":
                        fn_name = block.get("name", "")
                        fn_input = json.dumps(block.get("input", {}), ensure_ascii=False)
                        block_texts.append(f"\n<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {fn_input}}}\n</tool_call>")
                    elif b_type == "tool_result":
                        tool_id = block.get("tool_use_id", "tool")
                        res_content = block.get("content", "")
                        if isinstance(res_content, list):
                            res_content = " ".join([c.get("text", "") for c in res_content if isinstance(c, dict)])
                        is_err = " (Error)" if block.get("is_error") else ""
                        block_texts.append(f"Tool [{tool_id}]{is_err} Result:\n{res_content}")
                elif hasattr(block, "type"):
                    if block.type == "text" and block.text:
                        block_texts.append(block.text)
                    elif block.type == "thinking" and block.thinking:
                        block_texts.append(f"[Thinking: {block.thinking}]")
                    elif block.type == "tool_use":
                        fn_input = json.dumps(block.input or {}, ensure_ascii=False)
                        block_texts.append(f"\n<tool_call>\n{{\"name\": \"{block.name}\", \"arguments\": {fn_input}}}\n</tool_call>")
                    elif block.type == "tool_result":
                        res_content = block.content or ""
                        block_texts.append(f"Tool [{block.tool_use_id}] Result:\n{res_content}")

            combined_msg = " ".join(block_texts)
            history_messages.append(f"{role_label}: {combined_msg}")

    if history_messages:
        prompt_parts.append("Conversation History:\n" + "\n".join(history_messages))

    final_prompt = "\n\n".join(prompt_parts)

    from app.services.context_compressor import context_compressor
    final_prompt = context_compressor.compress_raw_prompt(final_prompt)

    # Определяем, включен ли режим рассуждений в Anthropic
    thinking_enabled = None
    if request.thinking and request.thinking.type == "enabled":
        thinking_enabled = True

    deepseek_req = DeepSeekChatRequest(
        prompt=final_prompt,
        chat_session_id=request.chat_session_id or request.session_id,
        model=request.model,
        thinking_enabled=thinking_enabled,
        stream=request.stream,
    )

    return deepseek_req, has_tools


def convert_deepseek_response_to_anthropic(
    resp: DeepSeekChatResponse,
    model: str,
    has_tools: bool = False,
) -> AnthropicMessagesResponse:
    """Преобразует синхронный ответ DeepSeek в AnthropicMessagesResponse."""
    content_blocks: List[AnthropicContentBlock] = []

    # 1. Обрабатываем вызовы инструментов
    clean_text = resp.content
    stop_reason = "end_turn"
    found_tool_calls = None

    if has_tools:
        clean_text, found_tool_calls = extract_tool_calls(resp.content)
        if found_tool_calls:
            stop_reason = "tool_use"
            for tc in found_tool_calls:
                try:
                    args_dict = json.loads(tc.function.arguments)
                except Exception:
                    args_dict = {"raw": tc.function.arguments}
                content_blocks.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=f"toolu_{uuid.uuid4().hex[:16]}",
                        name=tc.function.name,
                        input=args_dict,
                    )
                )

    # 2. Если вызовов инструментов не было, добавляем thinking и text блоки
    if not found_tool_calls:
        if resp.thinking:
            content_blocks.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=resp.thinking,
                )
            )
        if clean_text:
            content_blocks.append(AnthropicContentBlock(type="text", text=clean_text))

    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:20]}",
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=0,
            output_tokens=resp.token_usage or 0,
        ),
    )
