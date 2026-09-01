from typing import Annotated
from fastapi import Depends, Request
import httpx
from app.core.config import settings
from app.services.deepseek_client import DeepSeekClient


def get_http_client(request: Request) -> httpx.AsyncClient:
    if not hasattr(request.app.state, "http_client") or request.app.state.http_client is None:
        request.app.state.http_client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return request.app.state.http_client


def get_deepseek_client(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> DeepSeekClient:
    return DeepSeekClient(http_client=http_client)
