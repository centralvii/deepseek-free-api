import asyncio
import os
import sys
from typing import Optional
import httpx
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry
from app.schemas.chat import DeepSeekChatRequest
from app.services.session_manager import session_manager

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, write_through=True)
    except Exception:
        pass

console = Console()

prompt_style = Style.from_dict({
    "prompt": "#5fafff bold",
    "completion-menu.completion": "bg:#202020 #cccccc",
    "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
    "completion-menu.meta.completion": "bg:#202020 #888888",
    "completion-menu.meta.completion.current": "bg:#005f87 #aaaaaa italic",
    "bottom-toolbar": "bg:#1c1c1c #aaaaaa",
})


class MultiProviderCommandCompleter(Completer):
    """Динамическое автодополнение команд и моделей для всех провайдеров."""

    COMMANDS = {
        "/proxy": "Режим Proxy-монитора (прослушка и логирование запросов от Cline / Roo / Cursor)",
        "/login": "Войти через окно браузера и автоматически получить токен",
        "/provider": "Переключить активного провайдера (deepseek, qwen)",
        "/model": "Переключить модель LLM",
        "/token": "Установить Bearer токен (например, /token qwen <токен>)",
        "/think": "Режим рассуждений/мыслей (show / hide / off)",
        "/search": "Веб-поиск в реальном времени (on / off)",
        "/new": "Начать новый диалог (сбросить контекст)",
        "/sessions": "Показать список предыдущих диалогов",
        "/session": "Переключиться на диалог по его ID (например, /session <id>)",
        "/status": "Показать статус провайдеров, модель и ID сессии",
        "/clear": "Очистить экран терминала",
        "/help": "Показать список доступных команд",
        "/exit": "Выйти из консоли",
        "/quit": "Выйти из консоли",
    }

    SUBCOMMANDS = {
        "/proxy": {
            "start": "Запустить прокси-сервер и режим мониторинга агентов",
            "status": "Показать статус и эндпоинты для подключения Cline / Cursor",
        },
        "/login": {
            "deepseek": "Открыть браузер и войти в DeepSeek",
            "qwen": "Открыть браузер и войти в Qwen",
        },
        "/provider": {
            "deepseek": "DeepSeek (V4 Pro, V4 Flash, R1 Reasoner, V3)",
            "qwen": "Qwen Alibaba (Qwen 3.7 Plus, 3.8, 3.8-Coder, 3-Max, 3-Plus)",
        },
        "/think": {
            "show": "Включить рассуждения и показывать блок мыслей",
            "hide": "Включить рассуждения, но скрыть мысли (только ответ)",
            "on": "Включить рассуждения (показывать мысли)",
            "off": "Выключить рассуждения",
        },
        "/search": {
            "on": "Включить поиск в интернете",
            "off": "Выключить поиск",
        },
        "/model": {
            # DeepSeek
            "deepseek-v4-pro": "[DeepSeek] 1.6T MoE (49B act) рассуждения и сложный код",
            "deepseek-v4-flash": "[DeepSeek] 284B MoE сверхбыстрый чат",
            "deepseek-v4-flash-vision-exp": "[DeepSeek] Мультимодальная модель",
            "deepseek-reasoner": "[DeepSeek] R1 модель пошаговых рассуждений",
            "deepseek-chat": "[DeepSeek] V3 универсальный чат",
            "deepseek-search": "[DeepSeek] V3 с веб-поиском",
            # Qwen
            "qwen3.7-plus": "[Qwen] Актуальная веб-модель Qwen 3.7 Plus (Thinking)",
            "qwen-3.8": "[Qwen] Флагман 3-го поколения с рассуждениями",
            "qwen-3.8-coder": "[Qwen] Специализированная модель для сложного кодинга",
            "qwen-3-max": "[Qwen] Максимальная интеллектуальная мощность",
            "qwen-3-plus": "[Qwen] Быстрый универсальный ассистент",
            "qwen-3-flash": "[Qwen] Zero-Latency мгновенные ответы",
        },
        "/token": {
            "deepseek": "Установить токен для DeepSeek",
            "qwen": "Установить токен для Qwen",
        }
    }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parts = text.split()
        if len(parts) == 0:
            return

        if len(parts) == 1 and not text.endswith(" "):
            prefix = parts[0]
            for cmd, desc in self.COMMANDS.items():
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix), display_meta=desc)
        else:
            cmd = parts[0]
            if cmd in self.SUBCOMMANDS:
                sub_dict = self.SUBCOMMANDS[cmd]
                sub_prefix = parts[1] if len(parts) > 1 and not text.endswith(" ") else ""
                for sub_cmd, desc in sub_dict.items():
                    if sub_cmd.startswith(sub_prefix):
                        yield Completion(sub_cmd, start_position=-len(sub_prefix), display_meta=desc)


class MultiProviderCLI:
    def __init__(self):
        self.provider_id = "deepseek"
        self.model = "deepseek-v4-pro"
        self.thinking_mode = "show"  # "show" | "hide" | "off"
        self.search_enabled = False
        self.http_client: Optional[httpx.AsyncClient] = None
        self.session: Optional[PromptSession] = None

    async def init(self):
        self.http_client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        provider_registry.init_providers(self.http_client)
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=MultiProviderCommandCompleter(),
            style=prompt_style,
        )

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    @property
    def is_thinking_enabled(self) -> bool:
        return self.thinking_mode in ["show", "hide"]

    def get_active_session_id(self) -> Optional[str]:
        try:
            prov = provider_registry.get_provider(self.provider_id)
            return prov.get_current_session_id() or session_manager.get_current_session_id()
        except Exception:
            return session_manager.get_current_session_id()

    def get_bottom_toolbar(self):
        prov_name = provider_registry.get_provider(self.provider_id).display_name
        if self.thinking_mode == "show":
            think_str = "🧠 Мысли: ПОКАЗАТЬ"
        elif self.thinking_mode == "hide":
            think_str = "🧠 Мысли: СКРЫТЬ"
        else:
            think_str = "🧠 Мысли: ВЫКЛ"

        search_str = "🌐 Поиск: ВКЛ" if self.search_enabled else "🌐 Поиск: ВЫКЛ"
        sid = self.get_active_session_id()
        session_short = (sid[:8] + "...") if sid else "новая"
        return f" [{prov_name}] | [Модель: {self.model}] | [{think_str}] | [{search_str}] | [Сессия: {session_short}] "

    def print_banner(self):
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════════╗
║             Multi-LLM Reverse-Engineered Web CLI                 ║
║       DeepSeek V4/R1       •       Qwen 3.7 Plus / 3.8           ║
╚══════════════════════════════════════════════════════════════════╝[/bold cyan]
        """
        console.print(banner)
        self.print_status()
        console.print("[dim]Начните ввод сообщения или введите [bold]/[/bold] для вызова меню команд.[/dim]\n")

    def print_status(self):
        provider = provider_registry.get_provider(self.provider_id)
        is_auth = provider.is_authenticated()
        auth_status = f"[green]✓ Авторизован ({provider.display_name})[/green]" if is_auth else f"[bold red]✗ Нет токена ({provider.display_name})[/bold red]"
        session_id = self.get_active_session_id() or "[dim]не создана (будет создана при первом запросе)[/dim]"

        if self.thinking_mode == "show":
            think_label = "[green]ВКЛ (показывать блок мыслей)[/green]"
        elif self.thinking_mode == "hide":
            think_label = "[yellow]ВКЛ (скрывать мысли)[/yellow]"
        else:
            think_label = "[dim]ВЫКЛ[/dim]"

        tokens_info = []
        for p in provider_registry.list_providers():
            mark = "[green]✓[/green]" if p["authenticated"] else "[red]✗[/red]"
            active_mark = " [bold cyan](активен)[/bold cyan]" if p["id"] == self.provider_id else ""
            tokens_info.append(f"{mark} {p['name']}{active_mark}")

        status_table = (
            f"  • [bold]Провайдеры:[/bold] {' | '.join(tokens_info)}\n"
            f"  • [bold]Текущий статус:[/bold] {auth_status}\n"
            f"  • [bold]Выбранная модель:[/bold] [yellow]{self.model}[/yellow]\n"
            f"  • [bold]Thinking / Рассуждения:[/bold] {think_label}\n"
            f"  • [bold]Web Search:[/bold] {'[green]ВКЛ[/green]' if self.search_enabled else '[dim]ВЫКЛ[/dim]'}\n"
            f"  • [bold]ID Сессии:[/bold] [cyan]{session_id}[/cyan]"
        )
        console.print(Panel(status_table, title="[bold]Панель состояния[/bold]", border_style="blue"))

    def print_help(self):
        help_text = """
[bold cyan]Команды управления (поддерживается автодополнение по Tab):[/bold cyan]
  [bold yellow]/proxy[/bold yellow]                       - Перейти в режим Proxy-монитора (прослушка запросов от Cline/Roo)
  [bold yellow]/login [deepseek|qwen][/bold yellow]    - Автоматический вход через окно браузера (без ручного копирования)
  [bold yellow]/provider <deepseek|qwen>[/bold yellow] - Переключить активного провайдера
  [bold yellow]/model <name>[/bold yellow]              - Переключить модель (v4-pro, qwen3.7-plus, qwen-3.8-coder и др.)
  [bold yellow]/token [provider] <token>[/bold yellow]  - Установить Bearer токен вручную
  [bold yellow]/think [show|hide|off][/bold yellow]   - Режим мыслей: показывать, скрывать или выключить
  [bold yellow]/search [on|off][/bold yellow]           - Включить/выключить поиск в интернете
  [bold yellow]/new[/bold yellow]                       - Начать новый чат (сбросить контекст диалога)
  [bold yellow]/sessions[/bold yellow]                  - Список предыдущих сохраненных диалогов
  [bold yellow]/session <ID>[/bold yellow]              - Переключиться на существующий диалог
  [bold yellow]/status[/bold yellow]                    - Показать текущее состояние и статус токенов
  [bold yellow]/clear[/bold yellow]                     - Очистить экран терминала
  [bold yellow]/exit[/bold yellow] или [bold yellow]/quit[/bold yellow]            - Выйти из консоли
        """
        console.print(Panel(help_text, title="Справка", border_style="cyan"))

    async def start_background_server(self):
        """Запускает FastAPI Proxy сервер в фоне внутри текущего event loop."""
        if hasattr(self, "_server_task") and self._server_task and not self._server_task.done():
            return

        import uvicorn
        from app.main import app as fastapi_app

        config = uvicorn.Config(
            fastapi_app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.6)

    async def enter_proxy_mode(self):
        """Запускает интерактивный инспектор запросов от внешних AI-агентов (Cline, Cursor, Roo)."""
        console.print(f"[cyan]Инициализация Proxy-сервера на порту {settings.PORT}...[/cyan]")
        await self.start_background_server()

        proxy_banner = f"""
[bold cyan]╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                   🛡️ РЕЖИМ PROXY: МОНИТОРИНГ И ЛОГИРОВАНИЕ ЗАПРОСОВ АГЕНТОВ                     ║
║                                                                                                ║
║  • OpenAI Endpoint:    [bold yellow]http://127.0.0.1:{settings.PORT}/v1/chat/completions[/bold yellow]                               ║
║  • Anthropic Endpoint: [bold yellow]http://127.0.0.1:{settings.PORT}/v1/messages[/bold yellow]                                      ║
║  • API Key:            [bold green]любая строка (например, 'deepseek' или 'qwen')[/bold green]                        ║
║                                                                                                ║
║  [bold]Настройки подключения для Cline / Roo Code / Cursor:[/bold]                                      ║
║    API Provider: [yellow]OpenAI Compatible[/yellow] или [yellow]Anthropic[/yellow]                                         ║
║    Base URL:     [yellow]http://127.0.0.1:{settings.PORT}/v1[/yellow]                                                    ║
║    Model ID:     [yellow]deepseek-v4-pro[/yellow] | [yellow]deepseek-reasoner[/yellow] | [yellow]qwen-3.8-coder[/yellow]                          ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
[bold green]● Proxy-сервер активен и слушает входящие запросы.[/bold green]
[dim]Для возврата в режим интерактивного диалога нажмите [bold]Ctrl+C[/bold].[/dim]
"""
        console.print(proxy_banner)

        from app.services.proxy_logger import proxy_logger
        in_thinking = False

        def on_event(event: dict):
            nonlocal in_thinking
            e_type = event.get("type")

            if e_type == "request_start":
                in_thinking = False
                proto = event.get("protocol", "OpenAI")
                ep = event.get("endpoint", "")
                m = event.get("model", "")
                p = event.get("provider", "")
                toks = event.get("tokens", 0)
                msgs = event.get("messages_count", 0)
                tools = event.get("tools", [])
                t_str = f" | Tools ({len(tools)}): {', '.join(tools[:5])}{'...' if len(tools)>5 else ''}" if tools else " | Tools: нет"
                t_now = event.get("time", "")

                console.print(f"\n[bold magenta]┌── 📥 [{t_now}] Входящий запрос от агента ({event.get('user_agent', 'Agent')}) ──────────────────────[/bold magenta]")
                console.print(f"[bold magenta]│[/bold magenta] [bold cyan]Протокол:[/bold cyan] {proto} ({ep}) | [bold cyan]Модель:[/bold cyan] [yellow]{m}[/yellow] -> [green]{p}[/green]")
                console.print(f"[bold magenta]│[/bold magenta] [dim]Контекст: {msgs} сообщений (~{toks:,} токенов){t_str}[/dim]")
                console.print(f"[bold magenta]└── Потоковая генерация ответа ───────────────────────────────────────────────────────────[/bold magenta]")

            elif e_type == "thinking_chunk":
                if not in_thinking:
                    sys.stdout.write("\n\033[90m🧠 Рассуждения: ")
                    in_thinking = True
                sys.stdout.write(event.get("text", ""))
                sys.stdout.flush()

            elif e_type == "content_chunk":
                if in_thinking:
                    sys.stdout.write("\033[0m\n\n")
                    in_thinking = False
                text_chunk = event.get("text", "")
                # Не выводим служебную разметку протоколов инструментов (DSML, tool_call XML)
                if not any(tag in text_chunk for tag in ["<｜DSML｜", "<|DSML|", "<||DSML||", "<tool_call", "<invoke"]):
                    sys.stdout.write(text_chunk)
                    sys.stdout.flush()

            elif e_type == "tool_call":
                fn_name = event.get("tool_name", "")
                args = event.get("arguments", "")
                console.print(f"\n[bold yellow]🛠️  [Вызов инструмента][/bold yellow] [bold cyan]{fn_name}[/bold cyan]([dim]{args[:120]}{'...' if len(args)>120 else ''}[/dim])")

            elif e_type == "request_end":
                if in_thinking:
                    sys.stdout.write("\033[0m\n")
                    in_thinking = False
                status_code = event.get("status_code", 200)
                toks_out = event.get("tokens_out", 0)
                status_style = "bold green" if status_code == 200 else "bold red"
                console.print(f"\n[{status_style}]✓ Запрос завершен [{status_code}][/] | Сгенерировано: [cyan]{toks_out}[/cyan] токенов | Сервер готов к следующим запросам...\n")

        proxy_logger.subscribe(on_event)

        try:
            while True:
                await asyncio.sleep(0.5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            proxy_logger.unsubscribe(on_event)
            console.print("\n[yellow]Возврат в режим интерактивного чата.[/yellow]\n")
            self.print_status()

    def set_provider_and_default_model(self, pid: str):
        pid = pid.lower().strip()
        if pid == "qwen":
            self.provider_id = "qwen"
            self.model = "qwen3.7-plus"
            provider_registry.set_default_provider("qwen")
            console.print("[green]✓ Переключено на Qwen (по умолчанию модель qwen3.7-plus)[/green]")
        elif pid == "deepseek":
            self.provider_id = "deepseek"
            self.model = "deepseek-v4-pro"
            provider_registry.set_default_provider("deepseek")
            console.print("[green]✓ Переключено на DeepSeek (по умолчанию модель deepseek-v4-pro)[/green]")
        else:
            console.print(f"[red]Неизвестный провайдер:[/red] {pid}. Доступные: deepseek, qwen")

    async def handle_chat(self, user_input: str):
        provider = provider_registry.resolve_provider_for_model(self.model)

        if not provider.is_authenticated():
            console.print(
                f"[bold red]Ошибка:[/bold red] Учетные данные для {provider.display_name} не настроены.\n"
                f"Выполните автологин: [bold yellow]/login {provider.provider_id}[/bold yellow] или укажите токен: [bold yellow]/token {provider.provider_id} <токен>[/bold yellow]"
            )
            return

        req = DeepSeekChatRequest(
            prompt=user_input,
            chat_session_id=provider.get_current_session_id(),
            model=self.model,
            thinking_enabled=self.is_thinking_enabled,
            search_enabled=self.search_enabled,
            stream=True,
        )

        in_thinking = False
        in_content = False
        tokens_count = 0

        try:
            async for chunk in provider.stream_chat(req):
                if chunk.session_id:
                    provider.set_session_id(chunk.session_id)

                if chunk.token_usage:
                    tokens_count = chunk.token_usage

                # Обработка блока рассуждений (Thinking)
                if chunk.type == "thinking":
                    if self.thinking_mode == "show":
                        if not in_thinking:
                            console.print(f"\n[dim]╭─── 🧠 Рассуждения {provider.display_name} ─────────────────────────────────╮[/dim]")
                            sys.stdout.write("\033[90m")
                            in_thinking = True
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                    elif self.thinking_mode == "hide":
                        if not in_thinking:
                            sys.stdout.write(f"\r\033[90m🧠 {provider.display_name} рассуждает...\033[0m")
                            sys.stdout.flush()
                            in_thinking = True

                # Обработка основного ответа (Content)
                elif chunk.type == "content":
                    if in_thinking:
                        if self.thinking_mode == "show":
                            sys.stdout.write("\033[0m\n")
                            console.print("[dim]╰───────────────────────────────────────────────────────────────────────╯[/dim]\n")
                        elif self.thinking_mode == "hide":
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                        in_thinking = False

                    if not in_content:
                        console.print(f"[bold cyan]{provider.display_name}:[/bold cyan]")
                        in_content = True

                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()

            if in_thinking and self.thinking_mode == "show":
                sys.stdout.write("\033[0m\n")
                console.print("[dim]╰───────────────────────────────────────────────────────────────────────╯[/dim]")

            print("\n", flush=True)

            sid = provider.get_current_session_id() or session_manager.get_current_session_id() or "новая"
            info_str = f"[dim]Провайдер: {provider.display_name} | Модель: {self.model} | Сессия: {sid} | Использовано токенов: {tokens_count or 'N/A'}[/dim]\n"
            console.print(info_str)

        except Exception as e:
            console.print(f"\n[bold red]Ошибка выполнения запроса:[/bold red] {e}\n")

    async def run(self, auto_proxy: bool = False):
        await self.init()
        self.print_banner()

        if auto_proxy:
            await self.enter_proxy_mode()

        while True:
            try:
                user_input = await self.session.prompt_async(
                    [("class:prompt", "Вы > ")],
                    bottom_toolbar=self.get_bottom_toolbar,
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1].strip() if len(parts) > 1 else ""

                    if cmd in ["/exit", "/quit", "/q"]:
                        console.print("[cyan]До встречи![/cyan]")
                        break
                    elif cmd == "/help":
                        self.print_help()
                    elif cmd == "/status":
                        self.print_status()
                    elif cmd == "/clear":
                        os.system("cls" if os.name == "nt" else "clear")
                        self.print_banner()
                    elif cmd in ["/proxy", "/server"]:
                        await self.enter_proxy_mode()
                    elif cmd == "/login":
                        target_p = arg.lower().strip() if arg else self.provider_id
                        if target_p not in ["deepseek", "qwen"]:
                            target_p = self.provider_id
                        console.print(f"\n[bold cyan]🌐 Запуск браузера Chrome для авторизации в {target_p.upper()}...[/bold cyan]")
                        console.print("[dim]Войдите в аккаунт в открывшемся окне браузера. Токен будет перехвачен и сохранен автоматически.[/dim]\n")
                        from app.services.browser_auth import extract_token_via_browser
                        tok = await extract_token_via_browser(provider=target_p, headless=False, timeout_seconds=120)
                        if tok:
                            console.print(f"\n[bold green]✓ Токен для {target_p} успешно перехвачен и сохранен в credentials.json![/bold green]\n")
                        else:
                            console.print(f"\n[bold red]✗ Не удалось извлечь токен (таймаут или окно было закрыто).[/bold red]\n")
                        self.print_status()
                    elif cmd == "/provider":
                        if not arg:
                            console.print("[yellow]Использование:[/yellow] /provider <deepseek | qwen>")
                        else:
                            self.set_provider_and_default_model(arg)
                        self.print_status()
                    elif cmd == "/token":
                        token_parts = arg.split(maxsplit=1)
                        if len(token_parts) == 0 or not token_parts[0]:
                            console.print("[red]Использование:[/red] /token <токен> ИЛИ /token <provider> <токен>")
                        elif len(token_parts) == 1:
                            credentials_manager.save(token_parts[0], provider=self.provider_id)
                            console.print(f"[green]✓ Токен для {self.provider_id} успешно сохранен![/green]")
                            self.print_status()
                        else:
                            p_name, p_tok = token_parts[0].lower(), token_parts[1]
                            credentials_manager.save(p_tok, provider=p_name)
                            console.print(f"[green]✓ Токен для {p_name} успешно сохранен![/green]")
                            self.print_status()
                    elif cmd == "/new":
                        session_manager.reset_context()
                        try:
                            cur_p = provider_registry.get_provider(self.provider_id)
                            cur_p.reset_session()
                        except Exception:
                            pass
                        console.print("[green]✓ Начат новый диалог. Контекст сброшен.[/green]")
                        self.print_status()
                    elif cmd == "/sessions":
                        try:
                            cur_p = provider_registry.get_provider(self.provider_id)
                            sessions_list = await cur_p.list_sessions()
                            if not sessions_list:
                                console.print(f"[yellow]У провайдера {cur_p.display_name} нет сохраненных сессий или не удалось их получить.[/yellow]")
                            else:
                                console.print(f"\n[bold cyan]Список диалогов ({cur_p.display_name}):[/bold cyan]")
                                for idx, s in enumerate(sessions_list[:15], 1):
                                    s_id = s.get("id", "")
                                    s_title = s.get("title", "Без названия")
                                    is_curr = " [green](текущий)[/green]" if s_id == cur_p.get_current_session_id() else ""
                                    console.print(f"  {idx}. [yellow]{s_id}[/yellow] — [bold]{s_title}[/bold]{is_curr}")
                                console.print("[dim]Для переключения введите:[/dim] [bold yellow]/session <ID>[/bold yellow]\n")
                        except Exception as e:
                            console.print(f"[red]Ошибка при получении сессий:[/red] {e}")
                    elif cmd == "/session":
                        if not arg:
                            console.print("[yellow]Использование:[/yellow] /session <ID_сессии>")
                        else:
                            try:
                                cur_p = provider_registry.get_provider(self.provider_id)
                                cur_p.set_session_id(arg)
                                console.print(f"[green]✓ Переключено на сессию {arg} ({cur_p.display_name})[/green]")
                                self.print_status()
                            except Exception as e:
                                console.print(f"[red]Ошибка переключения сессии:[/red] {e}")
                    elif cmd == "/think":
                        arg_lower = arg.lower()
                        if arg_lower in ["show", "on"]:
                            self.thinking_mode = "show"
                            console.print("[green]✓ Режим рассуждений: мысли отображаются в отдельном блоке[/green]")
                        elif arg_lower in ["hide", "hidden"]:
                            self.thinking_mode = "hide"
                            console.print("[yellow]✓ Режим рассуждений: мысли скрыты (только финальный ответ)[/yellow]")
                        elif arg_lower == "off":
                            self.thinking_mode = "off"
                            console.print("[yellow]✓ Режим рассуждений ВЫКЛЮЧЕН[/yellow]")
                        else:
                            if self.thinking_mode == "off":
                                self.thinking_mode = "show"
                            elif self.thinking_mode == "show":
                                self.thinking_mode = "hide"
                            else:
                                self.thinking_mode = "off"
                        self.print_status()
                    elif cmd == "/search":
                        if arg.lower() == "on":
                            self.search_enabled = True
                            console.print("[green]✓ Веб-поиск ВКЛЮЧЕН[/green]")
                        elif arg.lower() == "off":
                            self.search_enabled = False
                            console.print("[yellow]✓ Веб-поиск ВЫКЛЮЧЕН[/yellow]")
                        else:
                            self.search_enabled = not self.search_enabled
                            console.print(f"Веб-поиск: {'[green]ВКЛ[/green]' if self.search_enabled else '[dim]ВЫКЛ[/dim]'}")
                        self.print_status()
                    elif cmd == "/model":
                        arg_clean = arg.lower().strip()
                        if not arg_clean:
                            console.print(f"[cyan]Текущая модель:[/cyan] {self.model}")
                        else:
                            # Автоматически определяем провайдера для выбранной модели
                            target_provider = provider_registry.resolve_provider_for_model(arg_clean)
                            self.provider_id = target_provider.provider_id
                            self.model = arg_clean
                            console.print(f"[green]✓ Выбрана модель: {self.model} (Провайдер: {target_provider.display_name})[/green]")
                        self.print_status()
                    else:
                        console.print(f"[red]Неизвестная команда:[/red] {cmd}. Введите [bold]/help[/bold].")
                    continue

                await self.handle_chat(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[cyan]Сессия завершена.[/cyan]")
                break

        await self.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek & Qwen Free API CLI & Proxy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--proxy", "-p",
        action="store_true",
        help="Сразу запустить режим Proxy-монитора (для ZCode, Cline, Cursor, Roo Code)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["deepseek", "qwen"],
        help="Выбрать провайдера по умолчанию (deepseek или qwen)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Выбрать модель по умолчанию (например, deepseek-v4-pro, qwen-3.8-coder)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Команда быстрого запуска (например, 'proxy' или 'server')",
    )

    args = parser.parse_args()

    cli = MultiProviderCLI()

    auto_proxy = args.proxy or (bool(args.command) and args.command.lower() in ["proxy", "server", "/proxy", "/server"])

    if args.provider:
        cli.set_provider_and_default_model(args.provider)
    if args.model:
        target_provider = provider_registry.resolve_provider_for_model(args.model)
        cli.provider_id = target_provider.provider_id
        cli.model = args.model

    asyncio.run(cli.run(auto_proxy=auto_proxy))


if __name__ == "__main__":
    main()
