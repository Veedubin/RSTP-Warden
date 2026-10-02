import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


@pytest.fixture
def app(db_with_user):
    app = create_app(WebSettings())

    @app.post("/echo-form")
    async def echo_form(request: Request):
        form = await request.form()
        return {"name": form.get("name")}

    @app.post("/echo-json")
    async def echo_json(request: Request):
        return await request.json()

    return app


def _csrf(client: TestClient) -> str:
    client.get("/login")
    return client.cookies.get("warden_csrf", "")


def test_form_body_token_is_accepted_and_body_reaches_handler(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post("/echo-form", data={"csrf_token": token, "name": "bob"})
    assert r.status_code == 200
    assert r.json() == {"name": "bob"}


def test_form_without_token_is_rejected(app):
    client = TestClient(app)
    _csrf(client)
    r = client.post("/echo-form", data={"name": "bob"})
    assert r.status_code == 403


def test_form_with_wrong_token_is_rejected(app):
    client = TestClient(app)
    _csrf(client)
    r = client.post("/echo-form", data={"csrf_token": "nope", "name": "bob"})
    assert r.status_code == 403


def test_json_post_with_header_token_keeps_body(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post("/echo-json", json={"pan": 0.5}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    assert r.json() == {"pan": 0.5}


def test_multipart_form_token_is_accepted(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post(
        "/echo-form",
        data={"csrf_token": token, "name": "multi"},
        files={"blob": ("b.txt", b"x")},
    )
    assert r.status_code == 200
    assert r.json() == {"name": "multi"}
