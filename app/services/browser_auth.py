import asyncio
import json
import logging
import os
import sys
from typing import Optional
from playwright.async_api import async_playwright

from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


def clean_deepseek_token(val: Optional[str]) -> Optional[str]:
    """Проверяет и очищает токен DeepSeek. Отсекает 'null', 'undefined' и пустые объекты."""
    if not val or not isinstance(val, str):
        return None
    
    t = val.strip()
    if t in ["null", "undefined", "None", "", "{}", "[]"]:
        return None
        
    # Если в localStorage лежит объект вида {"value": null, "__version": "0"}
    if t.startswith("{"):
        try:
            data = json.loads(t)
            v = data.get("value")
            if not v or str(v).lower() in ["null", "undefined", "none", ""]:
                return None
            t = str(v).strip()
        except Exception:
            return None

    if "value\":null" in t or "value\": null" in t:
        return None

    # Настоящий токен DeepSeek - это длинная строка (обычно JWT или токен сессии >= 30 символов)
    if len(t) >= 30 and (t.startswith("ey") or "." in t or len(t) >= 40):
        return t
        
    return None


def clean_qwen_token(val: Optional[str]) -> Optional[str]:
    """Проверяет и очищает сессионные Cookie/JWT для Qwen."""
    if not val or not isinstance(val, str):
        return None
    t = val.strip()
    if "token=eyJ" in t or (t.startswith("eyJ") and len(t) > 40):
        return t
    return None


async def extract_token_via_browser(
    provider: str = "deepseek",
    headless: bool = False,
    timeout_seconds: int = 300,
) -> Optional[str]:
    """
    Открывает системный браузер Google Chrome (или Edge),
    ждет реального входа пользователя в аккаунт и перехватывает токен авторизации.
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

        # 1. Перехват исходящих сетевых запросов
        async def on_request(request):
            nonlocal extracted_token
            req_url = request.url
            auth_hdr = request.headers.get("authorization") or request.headers.get("Authorization")

            if prov == "deepseek":
                if auth_hdr and "Bearer " in auth_hdr and "chat.deepseek.com" in req_url:
                    raw_tok = auth_hdr.replace("Bearer ", "").strip()
                    valid_tok = clean_deepseek_token(raw_tok)
                    if valid_tok and not extracted_token:
                        extracted_token = valid_tok
                        logger.info("Валидный токен DeepSeek перехвачен из сетевого запроса!")

            elif prov == "qwen":
                if auth_hdr and "Bearer " in auth_hdr and "chat.qwen.ai" in req_url:
                    raw_tok = auth_hdr.replace("Bearer ", "").strip()
                    valid_tok = clean_qwen_token(raw_tok)
                    if valid_tok and not extracted_token:
                        extracted_token = valid_tok
                        logger.info("Валидный токен Qwen перехвачен из сетевого запроса!")

        page.on("request", on_request)

        # 2. Мгновенный переход на страницу входа
        try:
            await page.goto(target_url, wait_until="commit", timeout=60000)
        except Exception as e:
            logger.warning(f"Навигация: {e}")

        # 3. Ожидание авторизации пользователя (до 5 минут)
        for sec in range(timeout_seconds):
            if extracted_token:
                break

            # Если пользователь закрыл вкладку/окно
            if page.is_closed():
                break

            try:
                if prov == "deepseek":
                    # Проверяем userToken в localStorage
                    ls_val = await page.evaluate("""() => {
                        try {
                            const raw = localStorage.getItem('userToken');
                            if (!raw) return null;
                            return raw;
                        } catch(e) {
                            return null;
                        }
                    }""")
                    valid_tok = clean_deepseek_token(ls_val)
                    if valid_tok:
                        extracted_token = valid_tok
                        logger.info("Валидный токен DeepSeek извлечен из localStorage!")
                        break

                elif prov == "qwen":
                    cookies = await context.cookies()
                    cookie_parts = []
                    found_jwt = None
                    for c in cookies:
                        if c.get("domain") and "qwen.ai" in c["domain"]:
                            cookie_parts.append(f"{c['name']}={c['value']}")
                            if c["name"] == "token" and len(c["value"]) > 30 and c["value"].startswith("eyJ"):
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
        print(f"\n[ОШИБКА] Не удалось получить токен (пользователь не вошел в аккаунт или окно закрыто).")
