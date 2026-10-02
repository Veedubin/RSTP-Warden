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


def _login_with_next(client: TestClient, next_value: str):
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    return client.post(
        "/login",
        data={
            "username": "admin",
            "password": "testpass123",
            "csrf_token": token,
            "next": next_value,
        },
        follow_redirects=False,
    )


def test_login_rejects_absolute_next(client):
    r = _login_with_next(client, "https://evil.example/phish")
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_login_rejects_protocol_relative_next(client):
    r = _login_with_next(client, "//evil.example/phish")
    assert r.headers["location"] == "/"


def test_login_keeps_local_next_with_query(client):
    r = _login_with_next(client, "/cameras?x=1")
    assert r.headers["location"] == "/cameras?x=1"


def test_login_page_redirects_authenticated_user_locally_only(client):
    _login_with_next(client, "/")
    r = client.get("/login?next=https://evil.example", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_html_redirect_carries_query_string(client):
    r = client.get("/cameras?x=1", headers={"Accept": "text/html"}, follow_redirects=False)
    assert r.headers["location"] == "/login?next=/cameras%3Fx%3D1"
