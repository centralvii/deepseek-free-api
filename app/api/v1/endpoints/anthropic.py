import json
import uuid
from typing import Annotated, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx

from app.api.deps import get_http_client
from app.providers.registry import provider_registry
from app.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from app.services.anthropic_converter import (
    convert_anthropic_request_to_deepseek,
    convert_deepseek_response_to_anthropic,
)
from app.services.tool_parser import extract_tool_calls

router = APIRouter(tags=["Anthropic"])


@router.post("/v1/messages", summary="Anthropic Messages API эндпоинт (/v1/messages с мульти-провайдерами)")
@router.post("/api/v1/messages", summary="Anthropic Messages API эндпоинт (/api/v1/messages с мульти-провайдерами)")
async def anthropic_messages(
    request: AnthropicMessagesRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поле messages обязательно и не может быть пустым."
        )

    provider = provider_registry.resolve_provider_for_model(request.model)
    deepseek_req, has_tools = convert_anthropic_request_to_deepseek(request)
    msg_id = f"msg_{uuid.uuid4().hex[:20]}"

    if request.stream:
        async def anthropic_sse_generator() -> AsyncGenerator[str, None]:
            start_event = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": request.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            }
            yield f"event: message_start\ndata: {json.dumps(start_event, ensure_ascii=False)}\n\n"

            block_index = 0
            in_thinking_block = False
            in_text_block = False
            accumulated_content = []

            async for chunk in provider.stream_chat(deepseek_req):
                if chunk.type == "thinking":
                    if not in_thinking_block:
                        cb_start = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                        in_thinking_block = True

                    cb_delta = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": chunk.text},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

                elif chunk.type == "content":
                    if in_thinking_block:
                        cb_stop = {"type": "content_block_stop", "index": block_index}
                        yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                        in_thinking_block = False
                        block_index += 1

                    accumulated_content.append(chunk.text)

                    if not has_tools:
                        if not in_text_block:
                            cb_start = {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                            in_text_block = True

                        cb_delta = {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": chunk.text},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

            if in_thinking_block:
                cb_stop = {"type": "content_block_stop", "index": block_index}
                yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                in_thinking_block = False
                block_index += 1

            stop_reason = "end_turn"
            full_text = "".join(accumulated_content)

            if has_tools:
                clean_text, tool_calls = extract_tool_calls(full_text)
                if clean_text:
                    cb_start = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                    cb_delta = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": clean_text},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"
                    cb_stop = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                    block_index += 1

                if tool_calls:
                    stop_reason = "tool_use"
                    for tc in tool_calls:
                        try:
                            args_dict = json.loads(tc.function.arguments)
                        except Exception:
                            args_dict = {"raw": tc.function.arguments}

                        tu_id = f"toolu_{uuid.uuid4().hex[:16]}"
                        cb_start = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tu_id,
                                "name": tc.function.name,
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"

                        cb_delta = {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(args_dict, ensure_ascii=False),
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

                        cb_stop = {"type": "content_block_stop", "index": block_index}
                        yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                        block_index += 1
                elif not clean_text and full_text:
                    cb_start = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                    cb_delta = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": full_text},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"
                    cb_stop = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                    block_index += 1
            else:
                if in_text_block:
                    cb_stop = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"

            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": len(accumulated_content)},
            }
            yield f"event: message_delta\ndata: {json.dumps(msg_delta, ensure_ascii=False)}\n\n"

            yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

        return StreamingResponse(
            anthropic_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        resp = await provider.send_message(deepseek_req)
        return convert_deepseek_response_to_anthropic(resp, model=request.model, has_tools=has_tools)
