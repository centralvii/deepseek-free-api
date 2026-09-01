# 🚀 Multi-LLM Web Reverse Proxy & Agent Gateway

<p align="center">
  <b>Высокопроизводительный асинхронный прокси-шлюз и интерактивная консоль для DeepSeek (V4 Pro / Flash / R1) и Qwen (3.7 Plus / 3.8 Coder)</b><br>
  Прямое взаимодействие с веб-версиями моделей без платных API-ключей, с поддержкой <b>Tool Use</b>, <b>1,000,000 токенов контекста</b> и интеграцией с AI-код-агентами (<b>Cline</b>, <b>Roo Code</b>, <b>Cursor</b>, <b>OpenCode</b>).
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenAI_API-Compatible-412991?style=flat-square&logo=openai" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/Anthropic_API-Compatible-d97706?style=flat-square&logo=anthropic" alt="Anthropic Compatible">
  <img src="https://img.shields.io/badge/Context-1M_Tokens-emerald?style=flat-square" alt="1M Context">
</p>

---

## 🌟 Ключевые возможности

- 🔓 **Бесплатный доступ без платных API-ключей**: Работает напрямую через официальные веб-сессии DeepSeek и Qwen.
- 🌐 **Мульти-провайдерная маршрутизация**: Единая точка входа для моделей DeepSeek и Qwen Alibaba с автоматическим выбором провайдера по имени модели.
- ⚡ **Мгновенный WASM Proof-of-Work (PoW)**: Встроенный солвер `DeepSeekHashV1` на WebAssembly решает криптографические челленджи менее чем за **50 мс**.
- 🔑 **Автологин через окно браузера (Playwright)**: Команды `/login deepseek` и `/login qwen` запускают системный браузер, перехватывают сессионные JWT/Bearer токены и сохраняют состояние сессии в `.browser_profile/`.
- 🧠 **1M Контекст и Интеллектуальный компрессор (~300k токенов)**:
  - Автоматическая защита от деградации внимания при длинных сессиях.
  - 100% сохранность системных промптов и определений инструментов (`Tools`).
  - Сохранение последних 12 сообщений без искажений и умное уплотнение старой середины диалога.
  - Автоматическое усечение гигантских дампов инструментов (`MAX_TOOL_OUTPUT_TOKENS = 25_000`).
- 🛠️ **Полноценная поддержка Tool Use (Function Calling)**: Преобразование и парсинг вызовов инструментов для автономных код-агентов (**Cline**, **Roo Code**, **Cursor**, **Claude Code**, **Aider**).
- 🔄 **Двойная совместимость API**:
  - `POST /v1/chat/completions` — на 100% совместим со спецификацией **OpenAI API** (включая `reasoning_content` и `stream=True`).
  - `POST /v1/messages` — совместим со спецификацией **Anthropic Messages API**.
- 🛡️ **Режим `/proxy` (Live Agent Monitor)**: Встроенный инспектор запросов прямо в терминале — отображает ход мыслей (Thinking), вызовы инструментов и метрики генерации в реальном времени от подключенных агентов.
- 🔒 **Выделенный порт `8317`**: Не конфликтует со стандартными портами локальной разработки (8000, 3000, 5000).

---

## 📋 Поддерживаемые модели

| Модель ID | Провайдер | Описание |
| :--- | :--- | :--- |
| `deepseek-v4-pro` | **DeepSeek** | 1.6T MoE (49B active) — флагман для сложного кода и архитектуры |
| `deepseek-v4-flash` | **DeepSeek** | 284B MoE — сверхбыстрая модель с минимальной задержкой |
| `deepseek-reasoner` | **DeepSeek** | DeepSeek-R1 — пошаговые рассуждения и логический анализ |
| `deepseek-chat` | **DeepSeek** | DeepSeek V3 — универсальный чат и решение задач |
| `qwen3.7-plus` | **Qwen** | Актуальная флагманская веб-модель Qwen 3.7 Plus с рассуждениями |
| `qwen-3.8-coder` | **Qwen** | Специализированная модель для разработки и рефакторинга |
| `qwen-3.8` | **Qwen** | Qwen 3.8 флагман общего назначения |
| `qwen-3-max` | **Qwen** | Максимальная вычислительная мощность Qwen |
| `claude-3-7-sonnet` / `claude-3-5-sonnet` | **DeepSeek** | Автоматическая трансляция запросов Anthropic в DeepSeek V4 |

---

## 📦 Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
git clone https://github.com/centralvii/deepseek-free-api.git
cd deepseek-free-api

# Установка Python-библиотек
pip install -r requirements.txt

# Установка браузера Chromium для автологина
playwright install chromium
```

> **Примечание:** Для решения PoW требуется установленный [Node.js](https://nodejs.org/) (версии 16+).

---

### 2. Авторизация (Автоматический вход)

Запустите консольный клиент:
```bash
python cli.py
```

В консоли введите команду авторизации:
```text
/login deepseek
```
*или для Qwen:*
```text
/login qwen
```

1. Откроется окно браузера с официальной страницей входа.
2. Войдите в свой аккаунт (через Google, GitHub, Email или Телефон).
3. Токен и сессия перехватятся **автоматически**, окно закроется, а токен сохранится в `credentials.json`!

*(Вы также можете ввести токен вручную командой `/token deepseek <токен>` или `/token qwen <токен>`)*.

---

## 🖥️ Использование интерактивной консоли (`cli.py`)

Интерактивный терминал поддерживает автодополнение команд по **Tab**, подсветку синтаксиса, стриминг мыслей модели и управление сессиями:

```text
/proxy              - Перейти в режим Proxy-монитора для код-агентов (Cline, Roo, Cursor)
/login [провайдер]  - Автоматический вход через окно браузера
/provider <id>      - Переключить активного провайдера (deepseek, qwen)
/model <name>       - Сменить модель (например, /model qwen-3.8-coder)
/think [show|hide]  - Показывать или скрывать блок рассуждений (Thinking)
/search [on|off]    - Включить веб-поиск в реальном времени
/new                - Начать новый чат со сбросом контекста
/sessions           - Список предыдущих диалогов
/session <ID>       - Переключиться на диалог по ID
/status             - Панель состояния провайдеров и токенов
/clear              - Очистить экран
/exit               - Выйти из консоли
```

---

## 🤖 Подключение AI-агентов (Cline, Roo Code, Cursor, OpenCode)

### Режим Proxy-инспектора в реальном времени

В консоли `cli.py` введите:
```text
/proxy
```
Сервер автоматически запустится на **`http://127.0.0.1:8317`** и перейдет в режим живого мониторинга всех запросов от агентов.

---

### Настройка в **Cline** / **Roo Code** (VS Code расширения)

1. Откройте настройки расширения **Cline** (шестеренка в правом верхнем углу).
2. Выберите **API Provider**: `OpenAI Compatible`.
3. Укажите:
   - **Base URL**: `http://127.0.0.1:8317/v1`
   - **API Key**: `deepseek` *(любое значение)*
   - **Model ID**: `deepseek-v4-pro` *(или `qwen-3.8-coder` / `deepseek-reasoner`)*
4. Нажмите **Done**. Теперь Cline пишет код, выполняет команды терминала и создает файлы через бесплатный шлюз!

*(При использовании Anthropic совместимого режима укажите Base URL `http://127.0.0.1:8317` и выберите провайдер `Anthropic`)*.

---

### Настройка в **Cursor**

1. Откройте **Settings** -> **Models** -> **OpenAI API Key**.
2. Включите галочку **Override OpenAI Base URL**.
3. Укажите: `http://127.0.0.1:8317/v1`
4. Добавьте модель: `deepseek-v4-pro` или `deepseek-reasoner`.

---

## 🐍 Использование через OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8317/v1",
    api_key="deepseek",  # Любая непустая строка
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "Ты опытный Python-разработчик."},
        {"role": "user", "content": "Напиши асинхронный генератор для чтения больших файлов."},
    ],
    stream=True,
)

for chunk in response:
    # Блок рассуждений модели (DeepSeek-R1 / Qwen Thinking)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # Основной сгенерированный ответ
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 🌐 Запуск отдельного API-сервера

Если вам требуется запустить сервер в качестве фоновой службы (без CLI):

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8317 --reload
```

- **Swagger UI (Интерактивная документация)**: [http://127.0.0.1:8317/docs](http://127.0.0.1:8317/docs)
- **ReDoc**: [http://127.0.0.1:8317/redoc](http://127.0.0.1:8317/redoc)
- **Health Check**: [http://127.0.0.1:8317/health](http://127.0.0.1:8317/health)

---

## ⚙️ Конфигурация (.env)

Вы можете настроить параметры в файле `.env`:

```env
# Параметры сервера
HOST=0.0.0.0
PORT=8317
DEBUG=false

# Параметры контекстного компрессора (1M окно токенов)
MAX_CONTEXT_TOKENS=300000
CONTEXT_COMPRESSION_ENABLED=true
RETAIN_RECENT_MESSAGES_COUNT=12
MAX_TOOL_OUTPUT_TOKENS=25000

# Сетевые таймауты
REQUEST_TIMEOUT=180.0
```

---

## 🧪 Запуск тестов

Проект полностью покрыт автоматическими тестами (мульти-провайдеры, PoW солвер, парсер SSE, Anthropic/OpenAI конвертеры, сжатие контекста):

```bash
python -m pytest tests/
```

---

## 📄 Лицензия

Проект распространяется под лицензией MIT. Создан исключительно в образовательных и исследовательских целях.
