from __future__ import annotations

import contextlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def browser_test_site(tmp_path: Path):
    (tmp_path / "index.html").write_text(
        """<!doctype html>
<html><head><title>Mana Browser Fixture</title></head>
<body>
  <form action="/complete.html">
    <label>Name <input name="name" aria-label="Name"></label>
    <label>Role <select name="role"><option>Developer</option><option>Reviewer</option></select></label>
    <button type="submit">Continue</button>
  </form>
  <a id="popup" href="/popup.html" target="_blank">Open details</a>
  <a id="download" href="/result.txt" download>Download result</a>
</body></html>""",
        encoding="utf-8",
    )
    (tmp_path / "complete.html").write_text("<title>Complete</title><h1>Complete</h1>", encoding="utf-8")
    (tmp_path / "popup.html").write_text("<title>Details</title><h1>Details</h1>", encoding="utf-8")
    (tmp_path / "result.txt").write_text("local browser result\n", encoding="utf-8")

    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(tmp_path), **kwargs
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError as exc:
        pytest.skip(f"Local HTTP sockets are unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_local_playwright_fixture_supports_browser_workflows(browser_test_site: str, tmp_path: Path) -> None:
    """Exercise the deterministic site used by browser-runtime integration coverage."""
    playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    manager = None
    try:
        manager = playwright.sync_playwright().start()
        browser = manager.chromium.launch(headless=True)
    except playwright.Error as exc:
        with contextlib.suppress(Exception):
            if manager is not None:
                manager.stop()
        pytest.skip(f"Playwright Chromium is unavailable: {exc}")

    try:
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(browser_test_site)
        assert page.title() == "Mana Browser Fixture"
        page.get_by_label("Name").fill("Mana")
        page.get_by_label("Role").select_option(label="Reviewer")

        with page.expect_popup() as popup_info:
            page.locator("#popup").click()
        popup = popup_info.value
        assert popup.get_by_role("heading").inner_text() == "Details"
        popup.close()

        with page.expect_download() as download_info:
            page.locator("#download").click()
        destination = tmp_path / "downloaded-result.txt"
        download_info.value.save_as(destination)
        assert destination.read_text(encoding="utf-8") == "local browser result\n"

        page.get_by_role("button", name="Continue").click()
        page.wait_for_url("**/complete.html?*")
        assert page.get_by_role("heading").inner_text() == "Complete"
        context.close()
    finally:
        browser.close()
        manager.stop()
