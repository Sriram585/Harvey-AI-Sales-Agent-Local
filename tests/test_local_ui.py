"""Tests for Phase 5: 100% offline Local Multi-Product AI Sales Agent dashboard UI."""

import os
import re
from pathlib import Path
from fastapi.testclient import TestClient

from sellbird.dashboard import app

WEB_DIR = Path(__file__).resolve().parent.parent / "sellbird" / "web"


def test_get_local_ui_returns_200():
    """Verify GET /local returns HTTP 200 and serves the offline agent dashboard."""
    client = TestClient(app, base_url="http://127.0.0.1:5555")
    resp = client.get("/local")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "HARVEY / SELLBIRD" in resp.text
    assert "LOCAL AGENT" in resp.text
    assert "Company Profile &amp; Product Portfolio" in resp.text
    assert "Prospect Pipeline &amp; Cross-Product Fit Matrix" in resp.text


def test_root_redirects_to_local_in_local_mode():
    """Verify GET / redirects to /local when LOCAL_AGENT_MODE is active."""
    os.environ["LOCAL_AGENT_MODE"] = "1"
    client = TestClient(app, base_url="http://127.0.0.1:5555")

    # Unfollowed redirect check
    resp_redirect = client.get("/", follow_redirects=False)
    assert resp_redirect.status_code in (302, 307)
    assert resp_redirect.headers.get("location") == "/local"

    # Followed redirect check
    resp_followed = client.get("/", follow_redirects=True)
    assert resp_followed.status_code == 200
    assert "HARVEY / SELLBIRD" in resp_followed.text


def test_static_assets_served_cleanly():
    """Verify local CSS, JS, and vendored fonts are served with correct headers."""
    client = TestClient(app, base_url="http://127.0.0.1:5555")

    # CSS
    resp_css = client.get("/static/local_agent.css")
    assert resp_css.status_code == 200
    assert "text/css" in resp_css.headers["content-type"]
    assert "--border-strong: #000000" in resp_css.text

    # JS
    resp_js = client.get("/static/local_agent.js")
    assert resp_js.status_code == 200
    assert "text/javascript" in resp_js.headers["content-type"]
    assert "api('/api/local/" in resp_js.text

    # Fonts
    resp_font1 = client.get("/static/fonts/geist.woff2")
    assert resp_font1.status_code == 200
    assert resp_font1.headers["content-type"] == "font/woff2"

    resp_font2 = client.get("/static/fonts/geist-mono.woff2")
    assert resp_font2.status_code == 200
    assert resp_font2.headers["content-type"] == "font/woff2"


def test_zero_external_cdn_or_urls_in_local_ui_files():
    """Verify local_agent.html, local_agent.css, and local_agent.js contain ZERO external URLs."""
    target_files = [
        WEB_DIR / "local_agent.html",
        WEB_DIR / "local_agent.css",
        WEB_DIR / "local_agent.js",
    ]

    url_pattern = re.compile(r"https?://([^\s\"\'<>]+)", re.IGNORECASE)
    forbidden_terms = ["cdn.", "googleapis", "jsdelivr", "unpkg", "cdnjs", "fonts.google"]

    for file_path in target_files:
        assert file_path.is_file(), f"File {file_path.name} must exist."
        content = file_path.read_text(encoding="utf-8")

        # Check for external URLs
        matches = url_pattern.findall(content)
        for url_match in matches:
            # Only loopback / localhost is allowed
            is_loopback = url_match.startswith("127.0.0.1") or url_match.startswith("localhost")
            assert is_loopback, f"Found forbidden external URL 'http(s)://{url_match}' in {file_path.name}"

        # Check for forbidden third-party CDN terms
        for term in forbidden_terms:
            assert term not in content.lower(), f"Found forbidden CDN reference '{term}' in {file_path.name}"
