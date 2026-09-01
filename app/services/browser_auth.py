import asyncio
import json
import logging
import os
from typing import Optional
from playwright.async_api import async_playwright

from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


async def extract_token_via_browser(
    provider: str = "deepseek",
    headless: bool = False,
    timeout_seconds: int = 120,
) -> Optional[str]:
    """
    Открывает системный браузер (Chrome / Edge) с постоянным профилем,
    позволяет пользователю войти в аккаунт и автоматически перехватывает
    и сохраняет рабочий токен авторизации для DeepSeek или Qwen.
    """
    prov = provider.lower().strip()
    profile_dir = os.path.abspath(f".browser_profile/{prov}")
    os.makedirs(profile_dir, exist_ok=True)

    target_url = "https://chat.deepseek.com" if prov == "deepseek" else "https://chat.qwen.ai"
    extracted_token = None

    async with async_playwright() as p:
        # Пробуем запустить Google Chrome, иначе Microsoft Edge
        context = None
        for channel in ["chrome", "msedge"]:
            try:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    channel=channel,
                    headless=headless,
                    viewport={"width": 1280, "height": 850},
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                )
                logger.info(f"Браузер ({channel}) успешно запущен для {prov}")
                break
            except Exception as e:
                logger.warning(f"Не удалось запустить канал '{channel}': {e}")

        if not context:
            logger.error("Не удалось запустить ни Chrome, ни Edge.")
            return None

        page = context.pages[0] if context.pages else await context.new_page()

        # 1. Перехват из сетевых запросов
        async def on_request(request):
            nonlocal extracted_token
            req_url = request.url
            auth_hdr = request.headers.get("authorization") or request.headers.get("Authorization")

            # DeepSeek Bearer токен
            if prov == "deepseek":
                if auth_hdr and "Bearer " in auth_hdr and "chat.deepseek.com" in req_url:
                    tok = auth_hdr.replace("Bearer ", "").strip()
                    if tok and len(tok) > 20 and not extracted_token:
                        extracted_token = tok
                        logger.info("Токен DeepSeek перехвачен из сетевого запроса!")

            # Qwen Bearer токен или Cookie
            elif prov == "qwen":
                if auth_hdr and "Bearer " in auth_hdr and "chat.qwen.ai" in req_url:
                    tok = auth_hdr.replace("Bearer ", "").strip()
                    if tok and len(tok) > 20 and not extracted_token:
                        extracted_token = tok
                        logger.info("Токен Qwen перехвачен из сетевого запроса!")

        page.on("request", on_request)

        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            logger.warning(f"Ошибка перехода на {target_url}: {e}")

        # 2. Периодический опрос localStorage и Cookies
        for _ in range(timeout_seconds):
            if extracted_token:
                break

            try:
                if prov == "deepseek":
                    # Проверка localStorage userToken
                    ls_val = await page.evaluate("""() => {
                        try {
                            const raw = localStorage.getItem('userToken');
                            if (!raw) return null;
                            const p = JSON.parse(raw);
                            return p.value || raw;
                        } catch(e) {
                            return localStorage.getItem('userToken') || localStorage.getItem('token');
                        }
                    }""")
                    if ls_val and len(str(ls_val)) > 20:
                        extracted_token = str(ls_val).strip()
                        logger.info("Токен DeepSeek извлечен из localStorage!")
                        break

                elif prov == "qwen":
                    cookies = await context.cookies()
                    cookie_parts = []
                    found_jwt = None
                    for c in cookies:
                        if c["domain"] and "qwen.ai" in c["domain"]:
                            cookie_parts.append(f"{c['name']}={c['value']}")
                            if c["name"] == "token" and len(c["value"]) > 30:
                                found_jwt = c["value"]

                    if cookie_parts and found_jwt:
                        extracted_token = "; ".join(cookie_parts)
                        logger.info("Сессионные Cookie и токен Qwen успешно извлечены!")
                        break

            except Exception:
                pass

            await asyncio.sleep(1)

        try:
            await context.close()
        except Exception:
            pass

    if extracted_token:
        credentials_manager.save(extracted_token, provider=prov)
        logger.info(f"✓ Токен для {prov} успешно сохранен в credentials.json!")
        return extracted_token

    return None


if __name__ == "__main__":
    import sys
    prov_arg = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    print(f"Запуск автологина для {prov_arg}...")
    res = asyncio.run(extract_token_via_browser(provider=prov_arg, headless=False, timeout_seconds=90))
    if res:
        print(f"\n[УСПЕХ] Токен для {prov_arg} успешно получен и сохранен в credentials.json!")
    else:
        print(f"\n[ОШИБКА] Не удалось получить токен. Попробуйте еще раз.")
