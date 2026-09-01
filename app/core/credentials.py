import json
import logging
from pathlib import Path
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class CredentialsManager:
    def __init__(self):
        self.project_file = settings.CREDENTIALS_PATH
        self.user_file = settings.USER_CREDENTIALS_PATH
        self.env_file = settings.PROJECT_ROOT / ".env"
        self._token: Optional[str] = None
        self.load()

    def _read_from_file(self, path: Path) -> Optional[str]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    token = data.get("token") or data.get("authorization") or data.get("ds_session_id")
                    if token:
                        token = token.strip()
                        if token.startswith("Bearer "):
                            token = token[7:].strip()
                        return token
            except Exception as e:
                logger.debug(f"Не удалось прочитать токен из {path}: {e}")
        return None

    def load(self) -> Optional[str]:
        token = self._read_from_file(self.project_file)
        if token:
            self._token = token
            return self._token

        token = self._read_from_file(self.user_file)
        if token:
            self._token = token
            return self._token

        if settings.DEEPSEEK_BEARER_TOKEN:
            t = settings.DEEPSEEK_BEARER_TOKEN.strip()
            if t.startswith("Bearer "):
                t = t[7:].strip()
            self._token = t
            return self._token

        return None

    def save(self, token: str) -> None:
        token = token.strip()
        if token.startswith("Bearer "):
            token = token[7:].strip()
        self._token = token

        payload = {"token": self._token}

        try:
            self.project_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.project_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Токен сохранен в {self.project_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить в {self.project_file}: {e}")

        try:
            self.user_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Токен сохранен в {self.user_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить в {self.user_file}: {e}")

        try:
            env_lines = []
            token_written = False
            if self.env_file.exists():
                with open(self.env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("DEEPSEEK_BEARER_TOKEN="):
                            env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{self._token}"\n')
                            token_written = True
                        else:
                            env_lines.append(line)

            if not token_written:
                env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{self._token}"\n')

            with open(self.env_file, "w", encoding="utf-8") as f:
                f.writelines(env_lines)
            logger.info(f"Токен записан в {self.env_file}")
        except Exception as e:
            logger.warning(f"Не удалось обновить {self.env_file}: {e}")

    @property
    def token(self) -> Optional[str]:
        if not self._token:
            self.load()
        return self._token

    @property
    def auth_header(self) -> str:
        t = self.token
        if not t:
            raise ValueError(
                "Учетные данные DeepSeek не установлены! Укажите токен через credentials.json, .env или команду входа."
            )
        return f"Bearer {t}"

    def is_authenticated(self) -> bool:
        return bool(self.token)


credentials_manager = CredentialsManager()
