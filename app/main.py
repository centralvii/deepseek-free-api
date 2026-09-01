import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.api.v1.api_router import api_router
from app.core.config import settings
from app.core.credentials import credentials_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Запуск {settings.APP_NAME} v{settings.APP_VERSION}")
    app.state.http_client = httpx.AsyncClient(
        timeout=settings.REQUEST_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )

    if credentials_manager.is_authenticated():
        logger.info("Учетные данные DeepSeek загружены и готовы к работе")
    else:
        logger.warning(
            "ВНИМАНИЕ: Учетные данные DeepSeek не найдены. Сохраните токен через /api/v1/auth/token или в credentials.json."
        )

    yield

    logger.info("Остановка приложения, закрытие сетевых соединений...")
    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", tags=["General"])
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "authenticated": credentials_manager.is_authenticated(),
        }

    @app.get("/health", tags=["General"])
    async def health():
        return {"status": "ok", "authenticated": credentials_manager.is_authenticated()}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
