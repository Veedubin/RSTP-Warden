import pytest
from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


@pytest.fixture
def client(db_with_user) -> TestClient:
    return TestClient(create_app(WebSettings()))


def test_html_request_without_session_redirects_to_login(client):
    r = client.get("/cameras", headers={"Accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/cameras"


def test_api_request_without_session_is_401_json(client):
    r = client.get("/cameras", headers={"Accept": "application/json"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"


def test_htmx_request_without_session_gets_hx_redirect(client):
    r = client.get("/cameras/x/status", headers={"HX-Request": "true", "Accept": "text/html"})
    assert r.status_code == 401
    assert r.headers["HX-Redirect"] == "/login"


def test_login_form_has_no_htmx_attributes(client):
    html = client.get("/login").text
    assert "hx-post" not in html


def test_successful_login_redirects(client):
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
