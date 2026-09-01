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
from app.schemas.chat import DeepSeekChatRequest
from app.services.deepseek_client import DeepSeekClient
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


class DeepSeekCommandCompleter(Completer):
    COMMANDS = {
        "/new": "Начать новый диалог (сбросить контекст)",
        "/think": "Режим рассуждений R1 (show / hide / off)",
        "/search": "Поиск в интернете (on / off)",
        "/model": "Переключить модель (chat, reasoner, search)",
        "/token": "Установить Bearer токен авторизации",
        "/status": "Показать статус, модель и ID сессии",
        "/clear": "Очистить экран терминала",
        "/help": "Показать список доступных команд",
        "/exit": "Выйти из консоли",
        "/quit": "Выйти из консоли",
    }

    SUBCOMMANDS = {
        "/think": {
            "show": "Включить R1 и показывать блок мыслей",
            "hide": "Включить R1, но скрыть мысли (только ответ)",
            "on": "Включить R1 (показывать мысли)",
            "off": "Выключить рассуждения (быстрый чат V3)",
        },
        "/search": {
            "on": "Включить поиск в интернете",
            "off": "Выключить поиск",
        },
        "/model": {
            "deepseek-v4-pro": "DeepSeek V4 Pro (1.6T MoE, рассуждения, код)",
            "deepseek-v4-flash": "DeepSeek V4 Flash (284B MoE, быстрая)",
            "deepseek-v4-flash-vision-exp": "DeepSeek V4 Vision (мультимодальная)",
            "deepseek-reasoner": "DeepSeek R1 (рассуждения)",
            "deepseek-chat": "DeepSeek V3 (быстрый чат)",
            "deepseek-search": "DeepSeek V3 (веб-поиск)",
            "v4-pro": "V4 Pro",
            "v4-flash": "V4 Flash",
            "v4-vision": "V4 Vision",
            "reasoner": "R1 Reasoner",
            "chat": "V3 Chat",
            "search": "V3 Search",
        },
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


class DeepSeekCLI:
    def __init__(self):
        self.model = "deepseek-chat"
        self.thinking_mode = "show"
        self.search_enabled = False
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[DeepSeekClient] = None
        self.session: Optional[PromptSession] = None

    async def init(self):
        self.http_client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        self.client = DeepSeekClient(self.http_client)
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=DeepSeekCommandCompleter(),
            style=prompt_style,
        )

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()

    @property
    def is_thinking_enabled(self) -> bool:
        return self.thinking_mode in ["show", "hide"]

    def get_bottom_toolbar(self):
        if self.thinking_mode == "show":
            think_str = "🧠 R1: ПОКАЗАТЬ МЫСЛИ"
        elif self.thinking_mode == "hide":
            think_str = "🧠 R1: СКРЫТЬ МЫСЛИ"
        else:
            think_str = "🧠 R1: ВЫКЛ"

        search_str = "🌐 Поиск: ВКЛ" if self.search_enabled else "🌐 Поиск: ВЫКЛ"
        sid = session_manager.get_current_session_id()
        session_short = (sid[:8] + "...") if sid else "новая сессия"
        return f" [Модель: {self.model}] | [{think_str}] | [{search_str}] | [Сессия: {session_short}] "

    def print_banner(self):
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════════╗
║              DeepSeek Reverse-Engineered Web CLI                 ║
║       Прямой доступ к chat.deepseek.com (без API ключей)         ║
╚══════════════════════════════════════════════════════════════════╝[/bold cyan]
        """
        console.print(banner)
        self.print_status()
        console.print("[dim]Начните ввод сообщения или введите [bold]/[/bold] для вызова меню команд.[/dim]\n")

    def print_status(self):
        auth_status = "[green]✓ Авторизован[/green]" if credentials_manager.is_authenticated() else "[bold red]✗ Нет токена[/bold red]"
        session_id = session_manager.get_current_session_id() or "[dim]не создана (будет создана при первом запросе)[/dim]"

        if self.thinking_mode == "show":
            think_label = "[green]ВКЛ (показывать блок мыслей)[/green]"
        elif self.thinking_mode == "hide":
            think_label = "[yellow]ВКЛ (скрывать мысли)[/yellow]"
        else:
            think_label = "[dim]ВЫКЛ[/dim]"

        status_table = (
            f"  • [bold]Статус auth:[/bold] {auth_status}\n"
            f"  • [bold]Модель:[/bold] [yellow]{self.model}[/yellow]\n"
            f"  • [bold]Deep Thinking (R1):[/bold] {think_label}\n"
            f"  • [bold]Web Search:[/bold] {'[green]ВКЛ[/green]' if self.search_enabled else '[dim]ВЫКЛ[/dim]'}\n"
            f"  • [bold]ID Сессии:[/bold] {session_id}"
        )
        console.print(Panel(status_table, title="[bold]Текущее состояние[/bold]", border_style="blue"))

    def print_help(self):
        help_text = """
[bold cyan]Команды управления (поддерживается автодополнение по Tab):[/bold cyan]
  [bold yellow]/help[/bold yellow]                    - Показать эту справку
  [bold yellow]/status[/bold yellow]                  - Показать текущее состояние, модель и сессию
  [bold yellow]/clear[/bold yellow]                   - Очистить экран терминала
  [bold yellow]/token <token>[/bold yellow]           - Установить/обновить Bearer токен авторизации
  [bold yellow]/new[/bold yellow]                     - Начать новый чат (сбросить контекст диалога)
  [bold yellow]/model <name>[/bold yellow]            - Переключить модель ([dim]chat, reasoner, search, pro, flash, vision[/dim])
  [bold yellow]/think [show|hide|off][/bold yellow] - Режим мыслей R1: показывать, скрывать или выключить
  [bold yellow]/search [on|off][/bold yellow]         - Включить/выключить поиск в интернете
  [bold yellow]/exit[/bold yellow] или [bold yellow]/quit[/bold yellow]          - Выйти из консоли
        """
        console.print(Panel(help_text, title="Справка", border_style="cyan"))

    async def handle_chat(self, user_input: str):
        if not credentials_manager.is_authenticated():
            console.print("[bold red]Ошибка:[/bold red] Укажите токен авторизации с помощью команды [bold]/token <ваш_токен>[/bold]")
            return

        req = DeepSeekChatRequest(
            prompt=user_input,
            model=self.model,
            thinking_enabled=self.is_thinking_enabled,
            search_enabled=self.search_enabled,
            stream=True,
        )

        in_thinking = False
        in_content = False
        tokens_count = 0

        try:
            async for chunk in self.client.stream_chat(req):
                if chunk.token_usage:
                    tokens_count = chunk.token_usage

                if chunk.type == "thinking":
                    if self.thinking_mode == "show":
                        if not in_thinking:
                            console.print("\n[dim]╭─── 🧠 Рассуждения DeepSeek-R1 ───────────────────────────────────────╮[/dim]")
                            sys.stdout.write("\033[90m")
                            in_thinking = True
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                    elif self.thinking_mode == "hide":
                        if not in_thinking:
                            sys.stdout.write("\r\033[90m🧠 DeepSeek-R1 рассуждает...\033[0m")
                            sys.stdout.flush()
                            in_thinking = True

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
                        console.print("[bold cyan]DeepSeek:[/bold cyan]")
                        in_content = True

                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()

            if in_thinking and self.thinking_mode == "show":
                sys.stdout.write("\033[0m\n")
                console.print("[dim]╰───────────────────────────────────────────────────────────────────────╯[/dim]")

            print("\n", flush=True)

            sid = session_manager.get_current_session_id() or ""
            info_str = f"[dim]Сессия: {sid[:8]}... | Использовано токенов: {tokens_count or 'N/A'}[/dim]\n"
            console.print(info_str)

        except Exception as e:
            console.print(f"\n[bold red]Ошибка выполнения запроса:[/bold red] {e}\n")

    async def run(self):
        await self.init()
        self.print_banner()

        if not credentials_manager.is_authenticated():
            console.print("[bold yellow]Внимание:[/bold yellow] Токен авторизации не найден.")
            try:
                user_token = await self.session.prompt_async("Вставьте ваш Bearer токен (chat.deepseek.com): ")
                user_token = user_token.strip()
                if user_token:
                    credentials_manager.save(user_token)
                    console.print("[green]✓ Токен успешно сохранен и активирован![/green]\n")
                    self.print_status()
            except (KeyboardInterrupt, EOFError):
                return

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
                    elif cmd == "/token":
                        if not arg:
                            console.print("[red]Использование:[/red] /token <bearer_token>")
                        else:
                            credentials_manager.save(arg)
                            console.print("[green]✓ Токен успешно сохранен и активирован![/green]")
                            self.print_status()
                    elif cmd == "/new":
                        session_manager.reset_context()
                        console.print("[green]✓ Начат новый диалог. Контекст сброшен.[/green]")
                        self.print_status()
                    elif cmd == "/think":
                        arg_lower = arg.lower()
                        if arg_lower in ["show", "on"]:
                            self.thinking_mode = "show"
                            self.model = "deepseek-reasoner"
                            console.print("[green]✓ Режим R1: рассуждения отображаются в отдельном блоке[/green]")
                        elif arg_lower in ["hide", "hidden"]:
                            self.thinking_mode = "hide"
                            self.model = "deepseek-reasoner"
                            console.print("[yellow]✓ Режим R1: рассуждения скрыты (только финальный ответ)[/yellow]")
                        elif arg_lower == "off":
                            self.thinking_mode = "off"
                            if self.model == "deepseek-reasoner":
                                self.model = "deepseek-chat"
                            console.print("[yellow]✓ Режим рассуждений ВЫКЛЮЧЕН (модель deepseek-chat)[/yellow]")
                        else:
                            if self.thinking_mode == "off":
                                self.thinking_mode = "show"
                                self.model = "deepseek-reasoner"
                            elif self.thinking_mode == "show":
                                self.thinking_mode = "hide"
                            else:
                                self.thinking_mode = "off"
                                self.model = "deepseek-chat"
                        self.print_status()
                    elif cmd == "/search":
                        if arg.lower() == "on":
                            self.search_enabled = True
                            self.model = "deepseek-search"
                            self.thinking_mode = "off"
                            console.print("[green]✓ Веб-поиск ВКЛЮЧЕН (модель deepseek-search)[/green]")
                        elif arg.lower() == "off":
                            self.search_enabled = False
                            if self.model == "deepseek-search":
                                self.model = "deepseek-chat"
                            console.print("[yellow]✓ Веб-поиск ВЫКЛЮЧЕН (модель deepseek-chat)[/yellow]")
                        else:
                            self.search_enabled = not self.search_enabled
                            if self.search_enabled:
                                self.model = "deepseek-search"
                                self.thinking_mode = "off"
                            elif self.model == "deepseek-search":
                                self.model = "deepseek-chat"
                            console.print(f"Веб-поиск: {'[green]ВКЛ[/green]' if self.search_enabled else '[dim]ВЫКЛ[/dim]'}")
                        self.print_status()
                    elif cmd == "/model":
                        arg_clean = arg.lower().strip()
                        if arg_clean in ["deepseek-v4-pro", "v4-pro", "v4", "pro"]:
                            self.model = "deepseek-v4-pro"
                            self.search_enabled = False
                            if self.thinking_mode == "off":
                                self.thinking_mode = "show"
                            console.print("[green]✓ Выбрана модель: DeepSeek V4 Pro 1.6T MoE (рассуждения + сложный код)[/green]")
                        elif arg_clean in ["deepseek-v4-flash", "v4-flash", "flash"]:
                            self.model = "deepseek-v4-flash"
                            self.search_enabled = False
                            self.thinking_mode = "off"
                            console.print("[green]✓ Выбрана модель: DeepSeek V4 Flash 284B MoE (высокая скорость)[/green]")
                        elif arg_clean in ["deepseek-v4-flash-vision-exp", "v4-vision", "vision"]:
                            self.model = "deepseek-v4-flash-vision-exp"
                            self.search_enabled = False
                            self.thinking_mode = "off"
                            console.print("[green]✓ Выбрана модель: DeepSeek V4 Flash Vision (мультимодальная)[/green]")
                        elif arg_clean in ["reasoner", "r1", "deepseek-reasoner"]:
                            self.model = "deepseek-reasoner"
                            self.search_enabled = False
                            if self.thinking_mode == "off":
                                self.thinking_mode = "show"
                            console.print("[green]✓ Выбрана модель: DeepSeek R1 (Reasoner)[/green]")
                        elif arg_clean in ["search", "deepseek-search"]:
                            self.model = "deepseek-search"
                            self.search_enabled = True
                            self.thinking_mode = "off"
                            console.print("[green]✓ Выбрана модель: DeepSeek V3 (Web Search)[/green]")
                        elif arg_clean in ["chat", "expert", "deepseek-chat", ""]:
                            self.model = "deepseek-chat"
                            self.search_enabled = False
                            self.thinking_mode = "off"
                            console.print("[green]✓ Выбрана модель: DeepSeek V3 (Chat)[/green]")
                        else:
                            self.model = arg
                            console.print(f"[green]✓ Установлена модель: {self.model}[/green]")
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
    cli = DeepSeekCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
