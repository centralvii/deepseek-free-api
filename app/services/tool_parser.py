import json
import re
import uuid
from typing import List, Optional, Tuple, Any, Dict
from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolCall, OpenAIToolCallFunction


def build_tool_system_prompt(tools: List[OpenAITool]) -> str:
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

    tools_json = json.dumps(tools_definitions, ensure_ascii=False, indent=2)

    prompt = f"""
# Available Tools
You have access to the following functions/tools to assist the user:

```json
{tools_json}
```

# Tool Call Instructions
When you need to call one or more tools, you MUST output each tool call inside a `<tool_call>` XML block with a valid JSON object containing `"name"` and `"arguments"`.

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
) -> str:
    prompt_parts = []

    if tools:
        tool_instruction = build_tool_system_prompt(tools)
        prompt_parts.append(tool_instruction)

    system_messages = []
    history_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if isinstance(content, list):
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

    return "\n\n".join(prompt_parts)


def extract_tool_calls(text: str) -> Tuple[str, List[OpenAIToolCall]]:
    tool_calls: List[OpenAIToolCall] = []
    clean_text = text

    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    for idx, match in enumerate(matches):
        raw_json = match.group(1).strip()
        try:
            data = json.loads(raw_json)
            name = data.get("name") or data.get("function")
            args = data.get("arguments", {})
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            else:
                args_str = str(args)

            if name:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls.append(
                    OpenAIToolCall(
                        id=call_id,
                        type="function",
                        function=OpenAIToolCallFunction(name=name, arguments=args_str),
                    )
                )
        except Exception:
            pass

    if tool_calls:
        clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return clean_text, tool_calls
