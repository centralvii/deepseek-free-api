import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry


@pytest.mark.asyncio
async def test_provider_registry_routing():
    providers = provider_registry.list_providers()
    provider_ids = [p["id"] for p in providers]
    assert "deepseek" in provider_ids
    assert "qwen" in provider_ids
    assert "glm" in provider_ids

    p_deepseek = provider_registry.resolve_provider_for_model("deepseek-v4-pro")
    assert p_deepseek.provider_id == "deepseek"

    p_qwen = provider_registry.resolve_provider_for_model("qwen-3.8-coder")
    assert p_qwen.provider_id == "qwen"

    p_glm = provider_registry.resolve_provider_for_model("glm-5.3")
    assert p_glm.provider_id == "glm"


@pytest.mark.asyncio
async def test_all_models_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        model_ids = [m["id"] for m in models]

        assert "deepseek-v4-pro" in model_ids
        assert "deepseek-v4-flash" in model_ids
        assert "deepseek-reasoner" in model_ids

        assert "qwen-3.8" in model_ids
        assert "qwen-3.8-coder" in model_ids
        assert "qwen-3-max" in model_ids

        assert "glm-5.3" in model_ids
        assert "glm-5-pro" in model_ids
        assert "glm-5-coder" in model_ids


@pytest.mark.asyncio
async def test_provider_switching_api():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_provider" in data
        assert len(data["providers"]) == 3

        sw_resp = await ac.post("/api/v1/providers/switch", json={"provider": "qwen"})
        assert sw_resp.status_code == 200
        assert sw_resp.json()["default_provider"] == "qwen"
        assert provider_registry.default_provider_id == "qwen"

        sw_resp2 = await ac.post("/api/v1/providers/switch", json={"provider": "deepseek"})
        assert sw_resp2.status_code == 200
        assert sw_resp2.json()["default_provider"] == "deepseek"


@pytest.mark.asyncio
async def test_multi_provider_credentials():
    credentials_manager.save("test_qwen_token_xyz", provider="qwen")
    credentials_manager.save("test_glm_token_xyz", provider="glm")

    assert credentials_manager.get_token("qwen") == "test_qwen_token_xyz"
    assert credentials_manager.get_token("glm") == "test_glm_token_xyz"
    assert credentials_manager.is_authenticated("qwen") is True
    assert credentials_manager.is_authenticated("glm") is True
