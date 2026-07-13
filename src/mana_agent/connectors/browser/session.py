from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mana_agent.config.settings import mana_home
from mana_agent.connectors.browser.approval import BrowserApprovalBinding, issue_approval
from mana_agent.connectors.browser.models import BrowserConfig


BLOCKED_PATTERNS = re.compile(r"\b(captcha|two[- ]factor|verification code|multi[- ]factor|access denied|security challenge)\b", re.I)
SENSITIVE_PATTERNS = re.compile(r"\b(pay|purchase|place order|publish|delete|remove account|accept|agree|sign up|create account|submit|send)\b", re.I)
SYSTEM_CHROMIUM_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)


def _chromium_executable(managed: str | Path) -> Path | None:
    managed_path = Path(managed)
    if managed_path.is_file():
        return managed_path
    return next((path for path in SYSTEM_CHROMIUM_CANDIDATES if path.is_file()), None)


class BrowserConnectorError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message); self.code = code


@dataclass
class BrowserSession:
    session_id: str
    playwright: Any
    browser: Any
    context: Any
    directory: Path
    pages: dict[str, Any] = field(default_factory=dict)
    active_tab: str = ""
    page_version: int = 0
    approvals: dict[str, Any] = field(default_factory=dict)
    pending_approvals: dict[str, Any] = field(default_factory=dict)


class BrowserSessionManager:
    """Own isolated Playwright contexts. Stateful calls are intentionally never cached."""

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def status() -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            executable = _chromium_executable(pw.chromium.executable_path)
            installed = executable is not None
            pw.stop()
            return {"ok": installed, "package_installed": True, "chromium_installed": installed, "executable": str(executable or "")}
        except ImportError:
            return {"ok": False, "package_installed": False, "chromium_installed": False, "error": "Install mana-agent[browser]."}
        except Exception as exc:
            return {"ok": False, "package_installed": True, "chromium_installed": False, "error": str(exc)}

    def _create(self, session_id: str, profile_name: str | None = None) -> BrowserSession:
        if not self.config.enabled:
            raise BrowserConnectorError("browser_disabled", "Browser tools are disabled in configuration.")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserConnectorError("browser_unavailable", "Playwright is unavailable. Install mana-agent[browser] and Chromium.") from exc
        pw = sync_playwright().start()
        base = mana_home() / "browser"
        base.mkdir(parents=True, exist_ok=True); os.chmod(base, 0o700)
        if profile_name:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", profile_name)
            directory = base / "profiles" / safe
        else:
            directory = Path(tempfile.mkdtemp(prefix=f"{session_id[:12]}-", dir=base))
        directory.mkdir(parents=True, exist_ok=True); os.chmod(directory, 0o700)
        try:
            executable = _chromium_executable(pw.chromium.executable_path)
            if executable is None:
                raise BrowserConnectorError("chromium_unavailable", "Chromium is unavailable. Run `python -m playwright install chromium` or install Google Chrome.")
            browser = pw.chromium.launch(headless=self.config.headless, executable_path=str(executable))
        except Exception as exc:
            pw.stop()
            raise BrowserConnectorError("chromium_unavailable", "Chromium is unavailable. Run `python -m playwright install chromium`.") from exc
        state_path = directory / "storage-state.json"
        context_args: dict[str, Any] = {"accept_downloads": True}
        if profile_name and state_path.is_file():
            context_args["storage_state"] = str(state_path)
        context = browser.new_context(**context_args)
        context.set_default_timeout(self.config.action_timeout_ms)
        context.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        session = BrowserSession(session_id, pw, browser, context, directory)
        self._sessions[session_id] = session
        return session

    def session(self, session_id: str, *, profile_name: str | None = None) -> BrowserSession:
        if not session_id.strip():
            raise BrowserConnectorError("decision_required", "A validated model-selected session_id is required; no default session was created.")
        with self._lock:
            return self._sessions.get(session_id) or self._create(session_id, profile_name)

    def _page(self, session: BrowserSession, tab_id: str | None = None) -> Any:
        selected = tab_id or session.active_tab
        if not selected or selected not in session.pages:
            raise BrowserConnectorError("page_required", "Open a page or select a valid tab first.")
        return session.pages[selected]

    def _register_page(self, session: BrowserSession, page: Any) -> str:
        tab_id = hashlib.sha256(str(id(page)).encode()).hexdigest()[:12]
        session.pages[tab_id] = page; session.active_tab = tab_id
        page.on("popup", lambda popup: self._register_page(session, popup))
        return tab_id

    def open(self, session_id: str, url: str, *, profile_name: str | None = None) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BrowserConnectorError("invalid_url", "browser_open requires an absolute HTTP(S) URL.")
        session = self.session(session_id, profile_name=profile_name)
        page = session.context.new_page(); tab_id = self._register_page(session, page)
        response = page.goto(url, wait_until="domcontentloaded")
        session.page_version += 1
        return {"ok": True, "tab_id": tab_id, "url": page.url, "title": page.title(), "status": response.status if response else None, "page_version": session.page_version}

    def inspect(self, session_id: str, *, tab_id: str | None = None) -> dict[str, Any]:
        session = self.session(session_id); page = self._page(session, tab_id)
        text = page.locator("body").inner_text(timeout=self.config.action_timeout_ms)[:50_000]
        blocked = BLOCKED_PATTERNS.search(text)
        if blocked:
            return {"ok": False, "error_code": "user_intervention_required", "message": f"Security control detected: {blocked.group(0)}. CAPTCHA/MFA/access controls will not be bypassed.", "url": page.url}
        elements = page.locator("a,button,input,select,textarea,[role=button]")
        controls = []
        for index in range(min(elements.count(), 300)):
            item = elements.nth(index)
            try:
                tag = item.evaluate("el => el.tagName.toLowerCase()")
                typ = item.get_attribute("type") or ""
                label = item.get_attribute("aria-label") or item.get_attribute("name") or item.inner_text(timeout=500) or item.get_attribute("placeholder") or ""
                if typ.lower() == "password": label = "[REDACTED PASSWORD FIELD]"
                controls.append({"ref": f"e{session.page_version}-{index}", "tag": tag, "type": typ, "label": str(label)[:300], "role": item.get_attribute("role")})
            except Exception:
                continue
        try:
            aria = page.locator("body").aria_snapshot(timeout=self.config.action_timeout_ms)
        except Exception:
            aria = ""
        return {"ok": True, "tab_id": session.active_tab, "url": page.url, "title": page.title(), "text": text, "controls": controls, "accessibility": str(aria)[:30_000], "forms": page.locator("form").count(), "page_version": session.page_version}

    def _locator(self, session: BrowserSession, page: Any, ref: str, observed: int | None) -> Any:
        if observed is None or observed != session.page_version:
            raise BrowserConnectorError("stale_reference", "Element references require the current observed_page_version. Inspect the page again.")
        match = re.fullmatch(r"e(\d+)-(\d+)", ref)
        if not match or int(match.group(1)) != session.page_version:
            raise BrowserConnectorError("invalid_reference", "Use an element ref returned by the latest browser_inspect call.")
        return page.locator("a,button,input,select,textarea,[role=button]").nth(int(match.group(2)))

    def act(self, session_id: str, action: str, *, tab_id: str | None = None, target: str = "", value: Any = None, observed_page_version: int | None = None, expected_origin: str | None = None, risk: str = "reversible", confirmation_required: bool = False, approval_token: str | None = None, timeout_ms: int | None = None) -> dict[str, Any]:
        session = self.session(session_id); page = self._page(session, tab_id)
        origin = f"{urlparse(page.url).scheme}://{urlparse(page.url).netloc}"
        if expected_origin and expected_origin != origin:
            raise BrowserConnectorError("origin_mismatch", "The live page origin differs from the model decision; inspect before continuing.")
        sensitive = confirmation_required or risk in {"sensitive", "irreversible"}
        locator = self._locator(session, page, target, observed_page_version) if target else None
        label = ""
        if locator is not None:
            label = (locator.get_attribute("aria-label") or locator.get_attribute("name") or locator.inner_text(timeout=500) or "")[:300]
        if action in {"click", "download"} and SENSITIVE_PATTERNS.search(label): sensitive = True
        binding = BrowserApprovalBinding(session_id=session_id, page_version=session.page_version, origin=origin, action=action, target=target, arguments={"value": "[REDACTED]" if action == "type" else value})
        if sensitive:
            approval = session.approvals.get(approval_token or "")
            if not approval or not approval.valid_for(binding):
                issued = issue_approval(binding); session.pending_approvals[issued.token] = issued
                return {"ok": False, "error_code": "confirmation_required", "confirmation_token": issued.token, "summary": f"Confirm {action} on {origin} target {label or target}", "expires_at": issued.expires_at.isoformat()}
            session.approvals.pop(approval_token or "", None)
        before = page.url
        download_payload = None
        if action == "download":
            with page.expect_download(timeout=timeout_ms or self.config.action_timeout_ms) as pending:
                locator.click(timeout=timeout_ms)
            download = pending.value
            suggested = Path(download.suggested_filename).name
            destination = session.directory / "downloads" / suggested
            destination.parent.mkdir(parents=True, exist_ok=True)
            download.save_as(str(destination))
            size = destination.stat().st_size
            if size > self.config.max_download_bytes:
                destination.unlink(missing_ok=True)
                raise BrowserConnectorError("download_too_large", "Downloaded file exceeded the configured size limit and was removed.")
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            download_payload = {"path": str(destination), "filename": suggested, "size": size, "sha256": digest}
        elif action == "click": locator.click(timeout=timeout_ms)
        elif action == "type": locator.fill(str(value or ""), timeout=timeout_ms)
        elif action == "select": locator.select_option(value, timeout=timeout_ms)
        elif action == "scroll": page.mouse.wheel(0, int(value or 600))
        elif action == "wait": page.wait_for_timeout(min(int(value or 1000), 30_000))
        elif action == "back": page.go_back(wait_until="domcontentloaded")
        else: raise BrowserConnectorError("unsupported_action", f"Unsupported browser action: {action}")
        if action in {"click", "type", "select", "back"}: session.page_version += 1
        return {"ok": True, "action": action, "url": page.url, "redirected": before != page.url, "page_version": session.page_version, "tab_id": session.active_tab, "download": download_payload}

    def approve(self, session_id: str | None, token: str) -> dict[str, Any]:
        """Promote a pending challenge only from the interactive chat command path."""
        candidates = [self._sessions.get(session_id)] if session_id else list(self._sessions.values())
        session = next((item for item in candidates if item is not None and token in item.pending_approvals), None)
        pending = session.pending_approvals.pop(token, None) if session else None
        if pending is None or pending.expires_at <= datetime.now(timezone.utc):
            raise BrowserConnectorError("invalid_confirmation", "Confirmation token is missing, expired, or belongs to another session.")
        session.approvals[token] = pending
        return {"ok": True, "approved": True, "token": token}

    def screenshot(self, session_id: str, *, tab_id: str | None = None, full_page: bool = True) -> dict[str, Any]:
        session = self.session(session_id); page = self._page(session, tab_id)
        path = session.directory / f"screenshot-{session.page_version}.png"; page.screenshot(path=str(path), full_page=full_page)
        return {"ok": True, "path": str(path), "url": page.url, "page_version": session.page_version}

    def upload(self, session_id: str, ref: str, path: str, *, observed_page_version: int, tab_id: str | None = None) -> dict[str, Any]:
        session = self.session(session_id); page = self._page(session, tab_id)
        resolved = Path(path).expanduser().resolve()
        roots = [Path(x).expanduser().resolve() for x in self.config.allowed_upload_roots]
        if roots and not any(resolved == root or root in resolved.parents for root in roots): raise BrowserConnectorError("upload_path_denied", "Upload path is outside configured allowed roots.")
        if not resolved.is_file(): raise BrowserConnectorError("upload_missing", "Upload file does not exist.")
        self._locator(session, page, ref, observed_page_version).set_input_files(str(resolved))
        session.page_version += 1
        return {"ok": True, "filename": resolved.name, "size": resolved.stat().st_size, "page_version": session.page_version}

    def tabs(self, session_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        return {"ok": True, "active_tab": session.active_tab, "tabs": [{"tab_id": key, "url": page.url, "title": page.title()} for key, page in session.pages.items()]}

    def check_links(self, session_id: str, *, tab_id: str | None = None, max_links: int = 50) -> dict[str, Any]:
        """Validate rendered HTTP(S) anchors without navigating the active page."""
        session = self.session(session_id); page = self._page(session, tab_id)
        anchors = page.locator("a[href]").evaluate_all(
            "els => els.map(a => ({href: a.href, text: (a.innerText || a.getAttribute('aria-label') || '').trim().slice(0, 160), visible: !!(a.offsetWidth || a.offsetHeight || a.getClientRects().length)}))"
        )
        unique: dict[str, dict[str, Any]] = {}
        for item in anchors:
            href = str(item.get("href") or "").split("#", 1)[0]
            if href.startswith(("http://", "https://")) and href not in unique:
                unique[href] = {**item, "href": href}
        results: list[dict[str, Any]] = []
        for item in list(unique.values())[: max(1, min(int(max_links), 100))]:
            try:
                response = session.context.request.get(
                    item["href"],
                    timeout=self.config.navigation_timeout_ms,
                    fail_on_status_code=False,
                )
                results.append({**item, "status": response.status, "ok": response.status < 400})
            except Exception as exc:
                results.append({**item, "status": None, "ok": False, "error": str(exc)[:300]})
        broken = [item for item in results if not item.get("ok")]
        return {"ok": True, "url": page.url, "checked": len(results), "broken": broken, "links": results, "page_version": session.page_version}

    def switch_tab(self, session_id: str, tab_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        if tab_id not in session.pages: raise BrowserConnectorError("invalid_tab", "Unknown browser tab id.")
        session.active_tab = tab_id; session.pages[tab_id].bring_to_front(); session.page_version += 1
        return {"ok": True, "tab_id": tab_id, "url": session.pages[tab_id].url, "page_version": session.page_version}

    def close(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.pop(session_id, None)
        if not session: return {"ok": True, "closed": False}
        if self.config.persistence == "named":
            state_path = session.directory / "storage-state.json"
            session.context.storage_state(path=str(state_path))
            os.chmod(state_path, 0o600)
        session.context.close(); session.browser.close(); session.playwright.stop()
        if self.config.persistence == "ephemeral": shutil.rmtree(session.directory, ignore_errors=True)
        return {"ok": True, "closed": True}


_MANAGER: BrowserSessionManager | None = None
def default_browser_manager(config: BrowserConfig | None = None) -> BrowserSessionManager:
    global _MANAGER
    if _MANAGER is None:
        if config is None:
            from mana_agent.config.settings import Settings
            settings = Settings()
            roots = [item.strip() for item in settings.mana_browser_upload_roots.split(os.pathsep) if item.strip()]
            config = BrowserConfig(
                enabled=settings.mana_browser_enabled,
                headless=settings.mana_browser_headless,
                action_timeout_ms=settings.mana_browser_timeout_seconds * 1000,
                navigation_timeout_ms=settings.mana_browser_timeout_seconds * 1000,
                max_download_bytes=settings.mana_browser_download_max_mb * 1024 * 1024,
                allowed_upload_roots=roots,
                persistence="named" if settings.mana_browser_persist_auth else "ephemeral",
            )
        _MANAGER = BrowserSessionManager(config)
    return _MANAGER
