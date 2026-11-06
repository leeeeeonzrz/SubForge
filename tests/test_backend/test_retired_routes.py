import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.api import free_models  # noqa: E402
from app.main import app  # noqa: E402


def test_local_catalog_status_retains_saved_provider_metadata(monkeypatch):
    profile = {
        "provider": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "vendor/chat-model",
        "api_key_configured": True,
    }
    monkeypatch.setattr(free_models, "get_llm_provider_status", lambda provider: profile)
    monkeypatch.setattr(free_models, "_active_task_id", None)
    monkeypatch.setattr(free_models, "_last_scan", None)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    response = client.get("/api/free-models/nvidia")
    assert response.status_code == 200
    assert response.json() == {**profile, "active_task_id": None, "last_scan": None}


def test_catalog_scan_requires_a_saved_api_key(monkeypatch):
    monkeypatch.setattr(
        free_models, "get_llm_provider_runtime_config",
        lambda provider: SimpleNamespace(api_key=""),
    )
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    response = client.post("/api/free-models/nvidia/scan")
    assert response.status_code == 400
    assert "API Key" in response.json()["detail"]


def test_foreign_browser_cannot_start_catalog_scan(monkeypatch):
    def forbidden_runtime_lookup(provider):
        raise AssertionError("Foreign requests must be blocked before credentials are read")

    monkeypatch.setattr(free_models, "get_llm_provider_runtime_config", forbidden_runtime_lookup)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    response = client.post(
        "/api/free-models/nvidia/scan",
        headers={"origin": "https://attacker.example", "sec-fetch-site": "cross-site"},
    )
    assert response.status_code == 403
