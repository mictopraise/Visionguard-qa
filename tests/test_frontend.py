from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_serves_browser_ui() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "VisionGuard QA" in response.text
    assert "Analyze videos" in response.text


def test_frontend_assets_are_served() -> None:
    css = client.get("/ui/styles.css")
    js = client.get("/ui/app.js")

    assert css.status_code == 200
    assert js.status_code == 200
    assert "issue-grid" in css.text
    assert "fetch('/compare'" in js.text
