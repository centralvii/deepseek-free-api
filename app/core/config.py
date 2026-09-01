from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "DeepSeek Web API Proxy"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    PROJECT_ROOT: Path = BASE_DIR
    CREDENTIALS_PATH: Path = BASE_DIR / "credentials.json"
    USER_CREDENTIALS_PATH: Path = Path.home() / ".deepseek" / "credentials.json"

    DEEPSEEK_BASE_URL: str = "https://chat.deepseek.com"
    DEEPSEEK_BEARER_TOKEN: str = Field(default="", description="Bearer token without 'Bearer ' prefix")
    
    CLIENT_BUNDLE_ID: str = "com.deepseek.chat"
    CLIENT_VERSION: str = "2.4.0"
    CLIENT_LOCALE: str = "ru"
    CLIENT_PLATFORM: str = "web"
    CLIENT_TIMEZONE_OFFSET: str = "10800"
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    
    REQUEST_TIMEOUT: float = 180.0


settings = Settings()
