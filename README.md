# DeepSeek Web API Proxy & Interactive CLI

Асинхронный сервис на FastAPI и интерактивная консоль для прямого взаимодействия с веб-версией **chat.deepseek.com** без использования платных API-ключей.

## Особенности и возможности

- 🚀 **Без платных API-ключей:** Работает через веб-сессию DeepSeek с авторизацией по токену.
- ⚡ **Встроенный Proof-of-Work (PoW):** Автоматическое решение задач `DeepSeekHashV1` через WASM-модуль за десятки миллисекунд (<50мс).
- 🧠 **Поддержка DeepSeek-R1:** Режим рассуждений с выводом цепочки мыслей (`thinking_enabled: true`).
- 🌐 **Поддержка Web Search:** Поиск актуальной информации в интернете (`search_enabled: true`).
- 💬 **Сохранение контекста диалога:** Автоматическое управление `chat_session_id` и `parent_message_id`.
- 🔄 **OpenAI и Anthropic совместимость:** Эндпоинты `/v1/chat/completions` и `/v1/messages` с поддержкой Tool-Use для подключения Cline, Cursor, Roo Code, Claude Dev.
- 💻 **Интерактивная консоль:** Удобный терминальный чат `python cli.py` с автодополнением по Tab и динамическим статус-баром.

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск интерактивной консоли

```bash
python cli.py
```

### 3. Запуск API сервера

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Документация Swagger UI: http://localhost:8000/docs
