import asyncio
import datetime
import logging
import sys
import time
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry
from app.schemas.chat import DeepSeekChatRequest
from app.services.banner import print_welcome_banner
from app.services.proxy_logger import proxy_logger

console = Console()


class DeepSeekCLI:
    def __init__(self):
        self.provider_id = provider_registry.default_provider_id
        # Выбираем дефолтную модель по текущему провайдеру
        if self.provider_id == "qwen":
            self.model = "qwen3.7-plus"
        else:
            self.model = "deepseek-v4-pro"
        self.session_id: Optional[str] = None
        self.thinking_enabled: bool = False
        self.search_enabled: bool = False
        self.stream_mode: bool = True
        self.available_models = {
            "deepseek": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner", "deepseek-chat"],
            "qwen": ["qwen3.7-plus", "qwen-3.8-coder", "qwen-3.8", "qwen3.8-max"],
        }

    def print_status(self):
        provider = provider_registry.get_provider(self.provider_id)
        p_name = provider.display_name if provider else self.provider_id
        auth_status = "[green]✓ Авторизован[/green]" if provider and provider.is_authenticated() else "[red]✗ Не авторизован[/red]"

        status_text = (
            f"[bold cyan]Провайдер:[/] [bold magenta]{p_name}[/] ({self.provider_id})\n"
            f"[bold cyan]Модель:[/] [bold yellow]{self.model}[/]\n"
            f"[bold cyan]Статус аккаунта:[/] {auth_status}\n"
            f"[bold cyan]Мысли (Thinking):[/] {'[green]ВКЛ[/green]' if self.thinking_enabled else '[dim]ВЫКЛ[/dim]'}\n"
            f"[bold cyan]Веб-поиск:[/] {'[green]ВКЛ[/green]' if self.search_enabled else '[dim]ВЫКЛ[/dim]'}\n"
            f"[bold cyan]Стриминг:[/] {'[green]ВКЛ[/green]' if self.stream_mode else '[dim]ВЫКЛ[/dim]'}\n"
            f"[bold cyan]Сессия:[/] [dim]{self.session_id or 'Новая (будет создана при первом запросе)'}[/dim]"
        )
        console.print(Panel(status_text, title="[bold]Текущее состояние[/bold]", border_style="cyan"))

    def print_help(self):
        help_table = Table(title="Доступные команды CLI", border_style="blue", show_header=True)
        help_table.add_column("Команда", style="bold yellow", width=22)
        help_table.add_column("Описание", style="white")

        help_table.add_row("/switch <provider>", "Переключить активного провайдера ([bold cyan]deepseek[/bold cyan] или [bold cyan]qwen[/bold cyan])")
        help_table.add_row("/model <name>", "Выбрать модель для генерации")
        help_table.add_row("/models", "Показать список моделей текущего провайдера")
        help_table.add_row("/proxy", "Перейти в режим Live Proxy Монитора (отслеживание запросов агентов)")
        help_table.add_row("/login [provider]", "Автологин через браузер (deepseek / qwen)")
        help_table.add_row("/token <p> <token>", "Вручную установить User Token для провайдера")
        help_table.add_row("/status", "Показать текущую модель, сессию и токены")
        help_table.add_row("/new", "Начать новый диалог (очистить историю сессии)")
        help_table.add_row("/think", "Переключить режим рассуждений (Thinking)")
        help_table.add_row("/search", "Переключить веб-поиск (Search)")
        help_table.add_row("/stream", "Переключить режим стриминга ответа")
        help_table.add_row("/exit, /quit", "Выйти из CLI")

        console.print(help_table)

    def print_models(self):
        provider = provider_registry.get_provider(self.provider_id)
        p_name = provider.display_name if provider else self.provider_id
        models = self.available_models.get(self.provider_id, [])

        table = Table(title=f"Доступные модели для {p_name}", border_style="green")
        table.add_column("ID модели", style="bold yellow")
        table.add_column("Статус", style="cyan")

        for m in models:
            is_cur = " (активна)" if m == self.model else ""
            table.add_row(m, f"[green]Доступна[/green]{is_cur}")

        console.print(table)
        console.print(f"[dim]Используйте: [bold]/model <ID>[/bold] для переключения[/dim]\n")

    def run_server_in_background(self):
        """Запускает FastAPI Proxy сервер в отдельном фоновом потоке."""
        import threading
        import uvicorn
        from app.main import app

        def _run():
            # Заглушаем избыточные uvicorn логи
            uvicorn.run(
                app,
                host=settings.HOST,
                port=settings.PORT,
                log_level="error",
                access_log=False,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.5)

    async def run_proxy_monitor(self):
        """Интерактивный экран Proxy режима для отслеживания запросов агентов в реальном времени."""
        console.clear()
        provider = provider_registry.get_provider(self.provider_id)
        p_name = provider.display_name if provider else self.provider_id

        header_text = (
            f"[bold green]⚡ РЕЖИМ ПРОКСИ-МОНИТОРА АКТИВИРОВАН[/bold green]\n"
            f"[cyan]Адрес для подключения агентов (ZCode, Cline, Cursor, Roo):[/cyan] [bold yellow]http://127.0.0.1:{settings.PORT}/v1[/bold yellow]\n"
            f"[cyan]Эндпоинт Anthropic (Claude Code):[/cyan] [bold yellow]http://127.0.0.1:{settings.PORT}/v1/messages[/bold yellow]\n"
            f"[cyan]Провайдер по умолчанию:[/cyan] [bold magenta]{p_name}[/] | [dim]Любой входящий API-ключ принимается[/dim]\n"
            f"[dim]Нажмите [bold red]Ctrl+C[/bold red] для возврата в интерактивный CLI чат.[/dim]"
        )
        console.print(Panel(header_text, border_style="green", expand=False))
        console.print("[dim]Ожидание входящих вызовов от ваших агентов...[/dim]\n")

        in_thinking = False

        def on_event(event: dict):
            nonlocal in_thinking
            e_type = event.get("type")

            if e_type == "request_start":
                in_thinking = False
                proto = event.get("protocol", "OpenAI")
                ep = event.get("endpoint", "/v1/chat/completions")
                m = event.get("model", "")
                p = event.get("provider", "")
                msgs = event.get("messages_count", 0)
                toks = event.get("tokens", 0)
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

        chat_req = DeepSeekChatRequest(
            prompt=user_input,
            chat_session_id=self.session_id,
            model=self.model,
            thinking_enabled=self.thinking_enabled,
            search_enabled=self.search_enabled,
            stream=self.stream_mode,
        )

        in_thinking = False
        full_content = []

        try:
            if self.stream_mode:
                console.print(f"\n[dim]{provider.display_name} отвечает...[/dim]")
                async for chunk in provider.stream_chat(chat_req):
                    if chunk.session_id and not self.session_id:
                        self.session_id = chunk.session_id

                    if chunk.type == "thinking":
                        if not in_thinking:
                            sys.stdout.write("\033[90m🧠 Рассуждения:\n")
                            in_thinking = True
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                    elif chunk.type == "content":
                        if in_thinking:
                            sys.stdout.write("\033[0m\n\n💬 Ответ:\n")
                            in_thinking = False
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                        full_content.append(chunk.text)

                if in_thinking:
                    sys.stdout.write("\033[0m\n")
                sys.stdout.write("\n\n")
            else:
                with console.status(f"[bold green]{provider.display_name} думает...", spinner="dots"):
                    resp = await provider.send_message(chat_req)

                if resp.session_id and not self.session_id:
                    self.session_id = resp.session_id

                if resp.thinking:
                    console.print(Panel(resp.thinking, title="🧠 Рассуждения", border_style="dim"))
                console.print(Panel(resp.content, title=f"💬 {provider.display_name}", border_style="green"))

        except Exception as e:
            console.print(f"[bold red]Ошибка генерации:[/] {e}")

    async def run(self):
        print_welcome_banner()
        self.print_status()

        # Автоматический запуск сервера в фоне
        self.run_server_in_background()
        console.print(f"[dim]Фоновый API-сервер запущен на [bold yellow]http://{settings.HOST}:{settings.PORT}[/bold yellow][/dim]")
        console.print("[dim]Введите [bold]/help[/bold] для справки или [bold]/proxy[/bold] для мониторинга агентов.[/dim]\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(f"[{self.provider_id}:{self.model}] > ").strip()
                )

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd_parts = user_input.split()
                    cmd = cmd_parts[0].lower()

                    if cmd in ["/exit", "/quit"]:
                        console.print("[yellow]До свидания![/yellow]")
                        break

                    elif cmd == "/help":
                        self.print_help()

                    elif cmd == "/status":
                        self.print_status()

                    elif cmd == "/switch":
                        if len(cmd_parts) > 1:
                            self.set_provider_and_default_model(cmd_parts[1])
                        else:
                            console.print("[yellow]Использование: /switch <deepseek|qwen>[/yellow]")

                    elif cmd == "/models":
                        self.print_models()

                    elif cmd == "/model":
                        if len(cmd_parts) > 1:
                            target_model = cmd_parts[1]
                            # Автоматически определяем провайдера для выбранной модели
                            resolved_p = provider_registry.resolve_provider_for_model(target_model)
                            self.provider_id = resolved_p.provider_id
                            self.model = target_model
                            console.print(f"[green]✓ Выбрана модель:[/] [yellow]{self.model}[/] (Провайдер: [magenta]{resolved_p.display_name}[/])")
                        else:
                            console.print("[yellow]Использование: /model <название модели>[/yellow]")

                    elif cmd == "/proxy":
                        await self.run_proxy_monitor()

                    elif cmd == "/new":
                        self.session_id = None
                        provider = provider_registry.get_provider(self.provider_id)
                        if provider:
                            provider.reset_session()
                        console.print("[green]✓ Сессия сброшена. Начат новый диалог.[/green]")

                    elif cmd == "/think":
                        self.thinking_enabled = not self.thinking_enabled
                        console.print(f"Режим рассуждений (Thinking): {'[green]ВКЛ[/green]' if self.thinking_enabled else '[red]ВЫКЛ[/red]'}")

                    elif cmd == "/search":
                        self.search_enabled = not self.search_enabled
                        console.print(f"Веб-поиск (Search): {'[green]ВКЛ[/green]' if self.search_enabled else '[red]ВЫКЛ[/red]'}")

                    elif cmd == "/stream":
                        self.stream_mode = not self.stream_mode
                        console.print(f"Режим стриминга: {'[green]ВКЛ[/green]' if self.stream_mode else '[red]ВЫКЛ[/red]'}")

                    elif cmd == "/login":
                        target_p = cmd_parts[1].lower() if len(cmd_parts) > 1 else self.provider_id
                        from app.services.browser_login import browser_login_service
                        console.print(f"[cyan]Запуск браузера для автологина в {target_p}...[/cyan]")
                        token = await browser_login_service.login(provider_id=target_p)
                        if token:
                            credentials_manager.set_token(target_p, token)
                            console.print(f"[bold green]✓ Автологин для {target_p} успешен! Токен сохранен.[/bold green]")
                        else:
                            console.print(f"[bold red]✗ Не удалось перехватить токен для {target_p}.[/bold red]")

                    elif cmd == "/token":
                        if len(cmd_parts) >= 3:
                            p_id = cmd_parts[1].lower()
                            tok = cmd_parts[2]
                            credentials_manager.set_token(p_id, tok)
                            console.print(f"[green]✓ Токен для {p_id} сохранен.[/green]")
                        else:
                            console.print("[yellow]Использование: /token <deepseek|qwen> <UserToken>[/yellow]")

                    else:
                        console.print(f"[red]Неизвестная команда:[/] {cmd}. Введите [bold]/help[/bold] для списка.")

                else:
                    await self.handle_chat(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]До свидания![/yellow]")
                break


def main():
    cli = DeepSeekCLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
