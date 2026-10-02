from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.alerts.manager import AlertManager, NotificationResult
from rtsp_warden.config import AlertsConfig, AppConfig
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


def _alerts_cfg() -> AlertsConfig:
    return AlertsConfig.model_validate(
        {
            "enabled": True,
            "notifiers": [{"name": "hook", "type": "webhook", "url": "http://127.0.0.1:9/x"}],
        }
    )


def test_manager_has_notifiers_without_start():
    mgr = AlertManager(_alerts_cfg())
    assert [n.name for n in mgr.notifiers] == ["hook"]


@pytest.fixture
def admin_client(db_with_user, monkeypatch) -> TestClient:
    cfg = AppConfig(cameras=[], alerts=_alerts_cfg())
    app = create_app(WebSettings(), cfg=cfg)

    async def fake_test(self, name):
        return NotificationResult(
            notifier_name=name, success=True, sent_at=datetime.now(tz=timezone.utc), http_status=200
        )

    monkeypatch.setattr(AlertManager, "test_notifier", fake_test)
    client = TestClient(app)
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post(
        "/login", data={"username": "admin", "password": "testpass123", "csrf_token": token}
    )
    return client


def test_test_button_is_post(admin_client):
    token = admin_client.cookies.get("warden_csrf", "")
    r = admin_client.post("/alerts/hook/test", headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert admin_client.get("/alerts/hook/test").status_code == 405
