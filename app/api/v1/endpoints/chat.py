import json
import logging
import time
import uuid
from typing import Annotated, AsyncGenerator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import httpx

logger = logging.getLogger(__name__)


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
    """
    Отправляет запрос в выбранный LLM провайдер (DeepSeek, Qwen, GLM).
    Поддерживает как стриминг (SSE), так и получение полного ответа сразу.
    """
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
    raw_req: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    """
    Эндпоинт, на 100% совместимый с форматом OpenAI API (/v1/chat/completions).
    Автоматически маршрутизирует модели:
    - deepseek-v4-pro, deepseek-reasoner, deepseek-chat -> DeepSeek
    - qwen3.7-plus, qwen-3.8, qwen-3.8-coder, qwen-3-max -> Qwen
    - Поддерживает Tool Use (Function Calling) и передачу reasoning_content
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Массив messages не может быть пустым"
        )

    provider = provider_registry.resolve_provider_for_model(request.model)

    from app.services.context_compressor import context_compressor, estimate_tokens
    from app.services.proxy_logger import proxy_logger

    # 1. Форматируем все сообщения и инструменты в единый контекстный промпт с учетом лимита провайдера
    provider_token_limit = context_compressor.get_limit_for_provider(provider.provider_id)
    compiled_prompt = format_messages_to_prompt(request.messages, request.tools, max_tokens=provider_token_limit)

    deepseek_req = DeepSeekChatRequest(
        prompt=compiled_prompt,
        chat_session_id=request.chat_session_id or request.session_id,
        model=request.model,
        stream=request.stream,
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    tools_names = [t.function.name for t in (request.tools or []) if t.function]
    ua = raw_req.headers.get("user-agent", "OpenAI Client")
    client_ip = raw_req.client.host if raw_req.client else "127.0.0.1"

    log_id = proxy_logger.log_request_start(
        protocol="OpenAI",
        endpoint="/v1/chat/completions",
        model=request.model,
        provider_name=provider.display_name,
        messages_count=len(request.messages),
        estimated_tokens=estimate_tokens(compiled_prompt),
        tools_names=tools_names,
        user_agent=ua,
        client_ip=client_ip,
    )

    # 2. Потоковый режим (Streaming)
    if request.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            first_chunk_sent = False
            accumulated_content = []
            has_tools = bool(request.tools)
            active_provider = provider

            try:
                async for chunk in active_provider.stream_chat(deepseek_req):
                    if chunk.type == "error":
                        raise HTTPException(status_code=400, detail=chunk.text)

                    if not first_chunk_sent and chunk.type in ["thinking", "content"]:
                        first_chunk = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                        )
                        yield f"data: {first_chunk.model_dump_json()}\n\n"
                        first_chunk_sent = True

                    # Мысли модели (DeepSeek-R1 / Qwen)
                    if chunk.type == "thinking":
                        proxy_logger.log_thinking_chunk(log_id, chunk.text)
                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(reasoning_content=chunk.text))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"

                    # Основной ответ
                    elif chunk.type == "content":
                        accumulated_content.append(chunk.text)
                        if not has_tools:
                            proxy_logger.log_content_chunk(log_id, chunk.text)
                            c = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=chunk.text))],
                            )
                            yield f"data: {c.model_dump_json()}\n\n"

                # Если были запрошены инструменты, проверяем сгенерированный текст на tool_calls
                finish_reason = "stop"
                full_text = "".join(accumulated_content)

                if has_tools:
                    clean_text, tool_calls = extract_tool_calls(full_text)
                    if tool_calls:
                        finish_reason = "tool_calls"
                        if clean_text:
                            proxy_logger.log_content_chunk(log_id, clean_text)
                        delta_tools = []
                        for idx, tc in enumerate(tool_calls):
                            proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)
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
                        if clean_text:
                            proxy_logger.log_content_chunk(log_id, clean_text)
                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=clean_text or full_text))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"

                # Завершающий чанк
                final_chunk = OpenAIChatCompletionChunk(
                    id=req_id,
                    model=request.model,
                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(), finish_reason=finish_reason)],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                proxy_logger.log_request_end(log_id, status_code=200, tokens_out=len(accumulated_content))

            except Exception as e:
                try:
                    active_provider.reset_session()
                except Exception:
                    pass
                err_detail = getattr(e, "detail", str(e))
                err_status = getattr(e, "status_code", 500)
                proxy_logger.log_request_end(log_id, status_code=err_status, error=str(err_detail))
                err_chunk = {
                    "error": {
                        "message": str(err_detail),
                        "type": "provider_error",
                        "code": err_status,
                    }
                }
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
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

    # 3. Синхронный режим (Non-streaming)
    else:
        try:
            resp = await provider.send_message(deepseek_req)

            clean_text = resp.content
            tool_calls: Optional[List[OpenAIToolCall]] = None
            finish_reason = "stop"

            if request.tools:
                clean_text, found_tool_calls = extract_tool_calls(resp.content)
                if found_tool_calls:
                    tool_calls = found_tool_calls
                    finish_reason = "tool_calls"
                    for tc in found_tool_calls:
                        proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)

            choice_message = OpenAIChoiceMessage(
                role="assistant",
                content=clean_text,
                reasoning_content=resp.thinking or None,
                tool_calls=tool_calls,
            )

            proxy_logger.log_request_end(log_id, status_code=200, tokens_out=resp.token_usage or 0)

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
        except Exception as e:
            proxy_logger.log_request_end(log_id, status_code=500, error=str(e))
            raise
