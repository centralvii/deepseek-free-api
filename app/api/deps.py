from typing import Annotated, Optional
from fastapi import Depends, Request
import httpx
from app.core.config import settings
from app.providers.base import BaseLLMProvider
from app.providers.registry import provider_registry
from app.services.deepseek_client import DeepSeekClient


def get_http_client(request: Request) -> httpx.AsyncClient:
    if not hasattr(request.app.state, "http_client") or request.app.state.http_client is None:
        request.app.state.http_client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    if not provider_registry._providers:
        provider_registry.init_providers(request.app.state.http_client)
    return request.app.state.http_client


def get_provider_registry(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> provider_registry:
    return provider_registry


def get_deepseek_client(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
) -> DeepSeekClient:
    return DeepSeekClient(http_client=http_client)
