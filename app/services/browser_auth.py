import asyncio
import json
import logging
import os
import sys
from typing import Optional
from playwright.async_api import async_playwright

from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


async def extract_token_via_browser(
    provider: str = "deepseek",
    headless: bool = False,
    timeout_seconds: int = 300,
) -> Optional[str]:
    """
    Открывает системный браузер Google Chrome (или Edge),
    мгновенно отображает страницу авторизации и непрерывно перехватывает
    токен авторизации в реальном времени.
    """
    prov = provider.lower().strip()
    profile_dir = os.path.abspath(".browser_profile")
    os.makedirs(profile_dir, exist_ok=True)
    state_file = os.path.join(profile_dir, f"{prov}_state.json")

    target_url = "https://chat.deepseek.com" if prov == "deepseek" else "https://chat.qwen.ai"
    extracted_token = None

    async with async_playwright() as p:
        browser = None
        for ch in ["chrome", "msedge"]:
            try:
                browser = await p.chromium.launch(
                    channel=ch,
                    headless=headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--start-maximized",
                    ],
                )
                logger.info(f"Браузер ({ch}) запущен.")
                break
            except Exception as e:
                logger.debug(f"Канал {ch} недоступен: {e}")

        if not browser:
            try:
                browser = await p.chromium.launch(headless=headless)
            except Exception as e:
                logger.error(f"Не удалось запустить браузер: {e}")
                return None

        context_kwargs = {
            "no_viewport": True,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
        }
        if os.path.exists(state_file):
            try:
                context_kwargs["storage_state"] = state_file
            except Exception:
                pass

        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()

        # 1. Мгновенный перехват заголовков Authorization из сетевого потока
        async def on_request(request):
            nonlocal extracted_token
            req_url = request.url
            auth_hdr = request.headers.get("authorization") or request.headers.get("Authorization")

            if prov == "deepseek":
                if auth_hdr and "Bearer " in auth_hdr and "chat.deepseek.com" in req_url:
                    tok = auth_hdr.replace("Bearer ", "").strip()
                    if tok and len(tok) > 20 and not extracted_token:
                        extracted_token = tok
                        logger.info("Токен DeepSeek перехвачен из сетевого запроса!")

            elif prov == "qwen":
                if auth_hdr and "Bearer " in auth_hdr and "chat.qwen.ai" in req_url:
                    tok = auth_hdr.replace("Bearer ", "").strip()
                    if tok and len(tok) > 20 and not extracted_token:
                        extracted_token = tok
                        logger.info("Токен Qwen перехвачен из сетевого запроса!")

        page.on("request", on_request)

        # 2. Мгновенный переход без блокировки загрузки (wait_until='commit')
        try:
            await page.goto(target_url, wait_until="commit", timeout=60000)
        except Exception as e:
            logger.warning(f"Навигация: {e}")

        # 3. Цикл непрерывного мониторинга входа
        for sec in range(timeout_seconds):
            if extracted_token:
                break

            # Если пользователь вручную закрыл страницу
            if page.is_closed():
                break

            try:
                if prov == "deepseek":
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
                        if c.get("domain") and "qwen.ai" in c["domain"]:
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

        if extracted_token:
            try:
                await context.storage_state(path=state_file)
            except Exception:
                pass

        try:
            await browser.close()
        except Exception:
            pass

    if extracted_token:
        credentials_manager.save(extracted_token, provider=prov)
        logger.info(f"✓ Токен для {prov} успешно сохранен в credentials.json!")
        return extracted_token

    return None


if __name__ == "__main__":
    prov_arg = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    print(f"Запуск окна браузера для {prov_arg}...")
    res = asyncio.run(extract_token_via_browser(provider=prov_arg, headless=False, timeout_seconds=300))
    if res:
        print(f"\n[УСПЕХ] Токен для {prov_arg} успешно получен и сохранен в credentials.json!")
    else:
        print(f"\n[ОШИБКА] Не удалось получить токен.")
