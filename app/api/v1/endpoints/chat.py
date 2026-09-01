import json
import time
import uuid
from typing import Annotated, AsyncGenerator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx

from app.api.deps import get_http_client
from app.providers.registry import provider_registry
from app.schemas.chat import DeepSeekChatRequest, DeepSeekChatResponse, StreamChunk
from app.schemas.openai import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIChoiceMessage,
    OpenAIChunkChoice,
    OpenAIDelta,
    OpenAIDeltaToolCall,
    OpenAIDeltaToolCallFunction,
    OpenAIToolCall,
    OpenAIUsage,
)
from app.services.tool_parser import extract_tool_calls, format_messages_to_prompt

router = APIRouter(tags=["Chat"])


@router.post("/api/v1/chat/send", summary="Отправить сообщение (Native, с автовыбором провайдера по модели)")
async def send_chat_message(
    request: DeepSeekChatRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    provider = provider_registry.resolve_provider_for_model(request.model)

    if request.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            async for chunk in provider.stream_chat(request):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await provider.send_message(request)


@router.post("/v1/chat/completions", summary="OpenAI-совместимый эндпоинт чата (с мульти-провайдерами, Tool-Use и Cline)")
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Массив messages не может быть пустым"
        )

    provider = provider_registry.resolve_provider_for_model(request.model)

    compiled_prompt = format_messages_to_prompt(request.messages, request.tools)

    deepseek_req = DeepSeekChatRequest(
        prompt=compiled_prompt,
        chat_session_id=request.chat_session_id,
        model=request.model,
        stream=request.stream,
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            first_chunk = OpenAIChatCompletionChunk(
                id=req_id,
                model=request.model,
                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            accumulated_content = []
            has_tools = bool(request.tools)

            async for chunk in provider.stream_chat(deepseek_req):
                if chunk.type == "thinking":
                    c = OpenAIChatCompletionChunk(
                        id=req_id,
                        model=request.model,
                        choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(reasoning_content=chunk.text))],
                    )
                    yield f"data: {c.model_dump_json()}\n\n"

                elif chunk.type == "content":
                    accumulated_content.append(chunk.text)
                    if not has_tools:
                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=chunk.text))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"

            finish_reason = "stop"
            full_text = "".join(accumulated_content)

            if has_tools:
                clean_text, tool_calls = extract_tool_calls(full_text)
                if tool_calls:
                    finish_reason = "tool_calls"
                    delta_tools = []
                    for idx, tc in enumerate(tool_calls):
                        delta_tools.append(
                            OpenAIDeltaToolCall(
                                index=idx,
                                id=tc.id,
                                type="function",
                                function=OpenAIDeltaToolCallFunction(
                                    name=tc.function.name,
                                    arguments=tc.function.arguments,
                                ),
                            )
                        )
                    c = OpenAIChatCompletionChunk(
                        id=req_id,
                        model=request.model,
                        choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=clean_text or None, tool_calls=delta_tools))],
                    )
                    yield f"data: {c.model_dump_json()}\n\n"
                else:
                    c = OpenAIChatCompletionChunk(
                        id=req_id,
                        model=request.model,
                        choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=full_text))],
                    )
                    yield f"data: {c.model_dump_json()}\n\n"

            final_chunk = OpenAIChatCompletionChunk(
                id=req_id,
                model=request.model,
                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(), finish_reason=finish_reason)],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        resp = await provider.send_message(deepseek_req)

        clean_text = resp.content
        tool_calls: Optional[List[OpenAIToolCall]] = None
        finish_reason = "stop"

        if request.tools:
            clean_text, found_tool_calls = extract_tool_calls(resp.content)
            if found_tool_calls:
                tool_calls = found_tool_calls
                finish_reason = "tool_calls"

        choice_message = OpenAIChoiceMessage(
            role="assistant",
            content=clean_text,
            reasoning_content=resp.thinking or None,
            tool_calls=tool_calls,
        )

        return OpenAIChatCompletionResponse(
            id=req_id,
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=choice_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=0,
                completion_tokens=resp.token_usage or 0,
                total_tokens=resp.token_usage or 0,
            ),
        )
