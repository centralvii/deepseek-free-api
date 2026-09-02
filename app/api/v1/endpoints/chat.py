import json
import logging
import re
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

INTENT_PAT = re.compile(
    r'(?:'
    r'изучу|исследую|посмотрю|проверю|гляну|разберу|проанализирую|прочитаю|открою|найду|загляну|ознакомлюсь|выполню|запущу|начну|'
    r'давайте\s+(?:изучим|посмотрим|проверим|исследуем|откроем|глянем)|'
    r'нужно\s+(?:изучить|посмотреть|проверить|исследовать|открыть|понять)|'
    r'let\s+me\s+(?:study|examine|investigate|analyze|review|search|scan|see|find|check|read|explore|inspect|run|look|implement)|'
    r'i\s*(?:will|\'ll|\s+need\s+to)\s+(?:study|examine|investigate|analyze|review|search|scan|see|find|check|read|explore|inspect|run|look|understand|implement)|'
    r'(?:next|first|now),?\s+(?:i\s+will|let\s+me)'
    r')'
    r'[^.!?\n]{0,120}'
    r'(?:файл|код|проект|директори|папк|api|структур|конфиг|скрипт|репозитори|file|code|dir|repo|output|struct|backend|frontend|project|folder|plan|service|parser)',
    re.IGNORECASE,
)

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
            accumulated_thinking = []
            has_tools = bool(request.tools)
            active_provider = provider

            try:
                async for chunk in active_provider.stream_chat(deepseek_req):
                    if chunk.type == "error":
                        raise HTTPException(status_code=400, detail=chunk.text)

                    # Мысли модели (DeepSeek-R1 / Qwen)
                    if chunk.type == "thinking":
                        proxy_logger.log_thinking_chunk(log_id, chunk.text)
                        accumulated_thinking.append(chunk.text)
                        # В обычном диалоге без инструментов стримим мысли немедленно
                        if not has_tools:
                            if not first_chunk_sent:
                                first_chunk = OpenAIChatCompletionChunk(
                                    id=req_id,
                                    model=request.model,
                                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                                )
                                yield f"data: {first_chunk.model_dump_json()}\n\n"
                                first_chunk_sent = True

                            c = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(reasoning_content=chunk.text))],
                            )
                            yield f"data: {c.model_dump_json()}\n\n"

                    # Основной ответ
                    elif chunk.type == "content":
                        accumulated_content.append(chunk.text)
                        # В обычном диалоге без инструментов стримим текст немедленно
                        if not has_tools:
                            if not first_chunk_sent:
                                first_chunk = OpenAIChatCompletionChunk(
                                    id=req_id,
                                    model=request.model,
                                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                                )
                                yield f"data: {first_chunk.model_dump_json()}\n\n"
                                first_chunk_sent = True

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

                    clean_prefix = re.split(r'<tool_calls?[^>]*>', full_text)[0].strip()

                    # Continuation Recovery: если модель заявила о намерении изучить файлы/выполнить код,
                    # или выдала поврежденный/не-JSON тег tool_call, запрашиваем строгое продолжение в JSON
                    if not tool_calls and (INTENT_PAT.search(full_text) or "<tool_call" in full_text):
                        logger.info("Обнаружено заявление намерения действия или невалидный tool_call. Запуск Continuation Recovery...")
                        try:
                            action_hint = clean_prefix[-150:] if len(clean_prefix) > 150 else clean_prefix
                            cont_req = DeepSeekChatRequest(
                                prompt=(
                                    f"{deepseek_req.prompt}\n\n"
                                    f"[Assistant response so far]\n{clean_prefix}\n\n"
                                    f"[STRICT INSTRUCTION: Execute the tool call for \"{action_hint}\" immediately. "
                                    f"Do NOT output raw code or file paths directly. "
                                    f"Output valid JSON inside <tool_call>: "
                                    f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{...}}}}\n</tool_call>]"
                                ),
                                chat_session_id=deepseek_req.chat_session_id,
                                model=deepseek_req.model,
                                stream=False,
                            )
                            cont_resp = await active_provider.send_message(cont_req)
                            cont_clean, cont_tools = extract_tool_calls(cont_resp.content)
                            if cont_tools:
                                tool_calls = cont_tools
                                if cont_clean:
                                    clean_text = (clean_text or clean_prefix) + "\n" + cont_clean
                                logger.info(f"✓ Continuation Recovery успешно извлек {len(cont_tools)} tool call(s)!")
                        except Exception as cont_err:
                            logger.warning(f"Ошибка Continuation Recovery: {cont_err}")

                    if tool_calls:
                        finish_reason = "tool_calls"
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

                        # Чанк 1: роль ассистента
                        first_chunk = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                        )
                        yield f"data: {first_chunk.model_dump_json()}\n\n"

                        # Чанк 2: вызов инструментов с content: None (строгий стандарт OpenAI, не сбивающий ai-sdk)
                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=None, tool_calls=delta_tools))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"
                    else:
                        # Если инструментов не обнаружено, отдаем накопленные рассуждения и ответ
                        first_chunk = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                        )
                        yield f"data: {first_chunk.model_dump_json()}\n\n"

                        if accumulated_thinking:
                            th_text = "".join(accumulated_thinking)
                            c = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(reasoning_content=th_text))],
                            )
                            yield f"data: {c.model_dump_json()}\n\n"

                        text_out = clean_text or full_text
                        if text_out:
                            proxy_logger.log_content_chunk(log_id, text_out)
                            c = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=text_out))],
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
            reasoning_to_return = resp.thinking or None

            if request.tools:
                clean_text, found_tool_calls = extract_tool_calls(resp.content)

                clean_prefix = re.split(r'<tool_calls?[^>]*>', resp.content)[0].strip()

                if not found_tool_calls and (INTENT_PAT.search(resp.content) or "<tool_call" in resp.content):
                    logger.info("Non-streaming: обнаружено намерение действия или невалидный tool_call. Запуск Continuation Recovery...")
                    try:
                        action_hint = clean_prefix[-150:] if len(clean_prefix) > 150 else clean_prefix
                        cont_req = DeepSeekChatRequest(
                            prompt=(
                                f"{deepseek_req.prompt}\n\n"
                                f"[Assistant response so far]\n{clean_prefix}\n\n"
                                f"[STRICT INSTRUCTION: Execute the tool call for \"{action_hint}\" immediately. "
                                f"Do NOT output raw code or file paths directly. "
                                f"Output valid JSON inside <tool_call>: "
                                f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{...}}}}\n</tool_call>]"
                            ),
                            chat_session_id=deepseek_req.chat_session_id,
                            model=deepseek_req.model,
                            stream=False,
                        )
                        cont_resp = await provider.send_message(cont_req)
                        cont_clean, cont_tools = extract_tool_calls(cont_resp.content)
                        if cont_tools:
                            found_tool_calls = cont_tools
                            if cont_clean:
                                clean_text = (clean_text or clean_prefix) + "\n" + cont_clean
                            logger.info(f"✓ Non-streaming Continuation Recovery успешно извлек {len(cont_tools)} tool call(s)!")
                    except Exception as cont_err:
                        logger.warning(f"Ошибка Continuation Recovery: {cont_err}")

                if found_tool_calls:
                    tool_calls = found_tool_calls
                    finish_reason = "tool_calls"
                    reasoning_to_return = None  # Не прикрепляем reasoning к tool_calls
                    clean_text = None  # В OpenAI tool_calls ход content должен быть null
                    for tc in found_tool_calls:
                        proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)

            choice_message = OpenAIChoiceMessage(
                role="assistant",
                content=clean_text,
                reasoning_content=reasoning_to_return,
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
