import pytest
from app.services.sse_parser import SSEParser, parse_sse_stream, parse_sse_lines


@pytest.mark.asyncio
async def test_sse_parser_user_example():
    """Тестирует парсер на точном примере SSE потока, предоставленном пользователем."""
    raw_events = [
        b"event: ready\r\ndata: {\"request_message_id\":1,\"response_message_id\":2,\"model_type\":\"expert\"}\r\n\r\n",
        b"event: update_session\r\ndata: {\"updated_at\":1788251660.8037179}\r\n\r\n",
        b"data: {\"v\":{\"response\":{\"message_id\":2,\"parent_id\":1,\"model\":\"\",\"role\":\"ASSISTANT\",\"thinking_enabled\":false,\"ban_edit\":false,\"ban_regenerate\":false,\"status\":\"WIP\",\"incomplete_message\":null,\"accumulated_token_usage\":0,\"feedback\":null,\"inserted_at\":1788251660.7872858,\"search_enabled\":false,\"fragments\":[{\"id\":2,\"type\":\"RESPONSE\",\"content\":\"\xD0\x9F\xD1\x80\xD0\xB8\",\"references\":[],\"stage_id\":1}],\"conversation_mode\":\"DEFAULT\",\"has_pending_fragment\":false,\"auto_continue\":false,\"search_triggered\":false}}}\r\n\r\n",
        b"data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"\xD0\xB2\xD0\xB5\xD1\x82\"}\r\n\r\n",
        b"data: {\"v\":\"!\"}\r\n\r\n",
        b"data: {\"v\":\" \xD0\xA7\"}\r\n\r\n",
        b"data: {\"v\":\"\xD0\xB5\xD0\xBC\"}\r\n\r\n",
        b"data: {\"v\":\" \xD0\xBC\xD0\xBE\xD0\xB3\xD1\x83\"}\r\n\r\n",
        b"data: {\"v\":\" \xD0\xBF\xD0\xBE\xD0\xBC\xD0\xBE\xD1\x87\xD1\x8c\"}\r\n\r\n",
        b"data: {\"v\":\"?\"}\r\n\r\n",
        b"data: {\"p\":\"response\",\"o\":\"BATCH\",\"v\":[{\"p\":\"accumulated_token_usage\",\"v\":46},{\"p\":\"quasi_status\",\"v\":\"FINISHED\"}]}\r\n\r\n",
        b"data: {\"p\":\"response/status\",\"o\":\"SET\",\"v\":\"FINISHED\"}\r\n\r\n",
        b"event: title\r\ndata: {\"content\":\"\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82\xD1\x81\xD1\x82\xD0\xB2\xD0\xB8\xD0\xB5\"}\r\n\r\n",
        b"event: close\r\ndata: {\"click_behavior\":\"none\",\"auto_resume\":false}\r\n\r\n",
    ]

    async def byte_generator():
        for b in raw_events:
            yield b

    collected_content = []
    collected_types = []
    token_usage = None

    async for chunk in parse_sse_stream(byte_generator(), session_id="test-session"):
        collected_types.append(chunk.type)
        if chunk.type == "content":
            collected_content.append(chunk.text)
        if chunk.token_usage:
            token_usage = chunk.token_usage

    full_text = "".join(collected_content)
    assert "Привет! Чем могу помочь?" in full_text
    assert token_usage == 46
    assert "status" in collected_types
    assert "title" in collected_types


@pytest.mark.asyncio
async def test_sse_parser_thinking_and_response():
    """Тестирует четкое разделение блоков рассуждений и ответа."""
    raw_events = [
        b'event: ready\r\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\r\n\r\n',
        b'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"First thought. "}]}}}\r\n\r\n',
        b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"Second thought."}\r\n\r\n',
        b'data: {"p":"response/fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"Final "}}\r\n\r\n',
        b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"answer."}\r\n\r\n',
        b'data: {"p":"response/status","o":"SET","v":"FINISHED"}\r\n\r\n',
        b"event: close\r\ndata: {}\r\n\r\n",
    ]

    async def gen():
        for b in raw_events:
            yield b

    thinking = []
    content = []
    async for chunk in parse_sse_stream(gen(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    assert "".join(thinking) == "First thought. Second thought."
    assert "".join(content) == "Final answer."


@pytest.mark.asyncio
async def test_sse_parser_batch_fragment_switch():
    """Тестирует переключение рассуждений на ответ через BATCH операцию."""
    raw_events = [
        b'event: ready\r\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\r\n\r\n',
        b'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"Thinking..."}]}}}\r\n\r\n',
        b'data: {"p":"response","o":"BATCH","v":[{"p":"fragments/0/status","o":"SET","v":"FINISHED"},{"p":"fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"Hello world"}}]}\r\n\r\n',
        b'data: {"v":"! How can I help?"}\r\n\r\n',
        b'data: {"p":"response/status","o":"SET","v":"FINISHED"}\r\n\r\n',
        b"event: close\r\ndata: {}\r\n\r\n",
    ]

    async def gen():
        for b in raw_events:
            yield b

    thinking = []
    content = []
    async for chunk in parse_sse_stream(gen(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    assert "".join(thinking) == "Thinking..."
    assert "".join(content) == "Hello world! How can I help?"


@pytest.mark.asyncio
async def test_sse_parser_real_deepseek_dump():
    """Тестирует парсер на реальном залогированном ответе от chat.deepseek.com с R1."""
    lines = [
        'event: ready',
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}',
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"Мы должны посчитать 2+2"}]}}}',
        'data: {"p":"response/fragments/0/content","o":"APPEND","v":". Ответ равен 4."}',
        'data: {"p":"response","o":"BATCH","v":[{"p":"fragments/0/status","o":"SET","v":"FINISHED"},{"p":"fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"4"}}]}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        'event: close',
        'data: {}',
    ]

    async def gen_lines():
        for l in lines:
            yield l

    thinking = []
    content = []
    async for chunk in parse_sse_lines(gen_lines(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    th = "".join(thinking)
    ct = "".join(content)
    assert "Мы должны" in th
    assert ct == "4"
