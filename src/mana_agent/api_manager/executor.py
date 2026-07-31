"""Controlled HTTP execution with DNS pinning, SSRF checks, and bounded responses."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import mimetypes
import socket
import ssl
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, quote, urljoin, urlsplit

from pydantic import Field

from mana_agent.api_manager.errors import (
    ApiCancelledError,
    ApiRateLimitError,
    ApiTimeoutError,
    BlockedHostError,
    DocumentationAuthorizationRequiredError,
    PermissionRequiredError,
    ResponseTooLargeError,
    SsrfPolicyViolationError,
    UpstreamApiError,
)
from mana_agent.api_manager.models import OperationRiskLevel, StrictModel
from mana_agent.api_manager.redaction import redact_mapping, redact_url
from mana_agent.api_manager.request_builder import BuiltApiRequest, RequestPreview


SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}


class NetworkAccessPolicy(StrictModel):
    allowed_schemes: tuple[str, ...] = ("https", "http")
    trusted_internal_hosts: tuple[str, ...] = ()
    trusted_internal_networks: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    maximum_redirects: int = Field(default=3, ge=0, le=10)
    maximum_response_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    allow_http: bool = True


class ApiExecutionResult(StrictModel):
    integration_id: str
    operation_id: str
    method: str
    redacted_url: str
    status_code: int
    headers: dict[str, str]
    content_type: str
    body_kind: str
    json_body: Any = None
    text_body: str = ""
    file_reference: str = ""
    binary_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    redirects: tuple[str, ...] = ()
    rate_limit: dict[str, str] = Field(default_factory=dict)
    attempts: int = 1
    executed: bool = True
    upstream_ok: bool


class ApprovalBroker(Protocol):
    def authorize(
        self,
        request: BuiltApiRequest,
        preview: RequestPreview,
        approval_reference: str,
    ) -> None: ...


class DenyMutationApprovalBroker:
    """Fail closed until a trusted UI attaches the repository approval flow."""

    def authorize(
        self,
        request: BuiltApiRequest,
        preview: RequestPreview,
        approval_reference: str,
    ) -> None:
        if not request.risk_level.mutating and not preview.approval_required:
            return
        raise PermissionRequiredError(
            "The API request requires approval through a trusted local client.",
            details={
                "permission_request_id": f"api_approval_{uuid.uuid4().hex}",
                "permission_scope": "api.request.execute",
                "preview": preview.model_dump(mode="json"),
            },
        )


@dataclass
class _PendingApproval:
    request: BuiltApiRequest
    preview: RequestPreview
    expires_at: datetime
    approved: bool = False


class PendingApiApprovalBroker:
    """Bind trusted approvals to the exact redacted request preview and session."""

    def __init__(self, *, ttl_seconds: int = 180) -> None:
        self.ttl_seconds = max(30, int(ttl_seconds))
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = threading.RLock()

    def authorize(
        self,
        request: BuiltApiRequest,
        preview: RequestPreview,
        approval_reference: str,
    ) -> None:
        if not request.risk_level.mutating and not preview.approval_required:
            return
        now = datetime.now(timezone.utc)
        with self._lock:
            self._expire(now)
            pending = self._pending.get(approval_reference)
            if (
                pending
                and pending.approved
                and _request_fingerprint(pending.request)
                == _request_fingerprint(request)
            ):
                self._pending.pop(approval_reference, None)
                return
            request_id = f"api_approval_{uuid.uuid4().hex}"
            self._pending[request_id] = _PendingApproval(
                request=request.model_copy(deep=True),
                preview=preview.model_copy(deep=True),
                expires_at=now + timedelta(seconds=self.ttl_seconds),
            )
        raise PermissionRequiredError(
            "The API request is waiting for a trusted local approval.",
            details={
                "permission_request_id": request_id,
                "permission_scope": "api.request.execute",
                "preview": preview.model_dump(mode="json"),
                "session_id": request.session_id,
                "api_approval": True,
                "expires_at": self._pending[request_id].expires_at.isoformat(),
            },
        )

    def approve(self, request_id: str, *, session_id: str, client_type: str) -> tuple[BuiltApiRequest, RequestPreview]:
        if client_type not in {"local_cli", "tui", "dashboard"}:
            raise PermissionError("API approvals must come from a trusted local client.")
        now = datetime.now(timezone.utc)
        with self._lock:
            self._expire(now)
            pending = self._pending.get(request_id)
            if pending is None:
                raise LookupError("No API request is waiting for that approval.")
            if pending.request.session_id != session_id:
                raise PermissionError("The API approval belongs to a different session.")
            pending.approved = True
            return pending.request.model_copy(deep=True), pending.preview.model_copy(deep=True)

    def deny(self, request_id: str, *, session_id: str, client_type: str) -> None:
        if client_type not in {"local_cli", "tui", "dashboard"}:
            raise PermissionError("API approvals must come from a trusted local client.")
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                raise LookupError("No API request is waiting for that approval.")
            if pending.request.session_id != session_id:
                raise PermissionError("The API approval belongs to a different session.")
            self._pending.pop(request_id, None)

    def _expire(self, now: datetime) -> None:
        expired = [key for key, value in self._pending.items() if value.expires_at <= now]
        for key in expired:
            self._pending.pop(key, None)


class ApiEventSink(Protocol):
    def __call__(self, event_type: str, payload: dict[str, Any]) -> None: ...


class HttpTransport(Protocol):
    def send(
        self,
        method: str,
        url: str,
        *,
        resolved_ip: str,
        headers: dict[str, str],
        body: bytes | None,
        connect_timeout: float,
        read_timeout: float,
        maximum_bytes: int,
        cancellation: threading.Event | None,
    ) -> "_RawResponse": ...


class PinnedHttpTransport:
    def send(
        self,
        method: str,
        url: str,
        *,
        resolved_ip: str,
        headers: dict[str, str],
        body: bytes | None,
        connect_timeout: float,
        read_timeout: float,
        maximum_bytes: int,
        cancellation: threading.Event | None,
    ) -> "_RawResponse":
        return _pinned_request(
            method,
            url,
            resolved_ip=resolved_ip,
            headers=headers,
            body=body,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            maximum_bytes=maximum_bytes,
            cancellation=cancellation,
        )


class ApiExecutor:
    def __init__(
        self,
        *,
        network_policy: NetworkAccessPolicy | None = None,
        approval_broker: ApprovalBroker | None = None,
        event_sink: ApiEventSink | None = None,
        artifact_directory: Path | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.network_policy = network_policy or NetworkAccessPolicy()
        self.approval_broker = approval_broker or DenyMutationApprovalBroker()
        self.event_sink = event_sink
        self.artifact_directory = artifact_directory
        self.transport = transport or PinnedHttpTransport()
        self.sleep = sleep

    def execute(
        self,
        request: BuiltApiRequest,
        *,
        preview: RequestPreview,
        approval_reference: str = "",
        cancellation: threading.Event | None = None,
    ) -> ApiExecutionResult:
        if request.risk_level.mutating or preview.approval_required:
            self._emit(
                "api.approval.required",
                request,
                preview=preview.model_dump(mode="json"),
            )
        self.approval_broker.authorize(request, preview, approval_reference)
        attempts = max(1, request.retry_maximum_attempts)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            self._check_cancelled(cancellation)
            self._emit("api.call.started", request, attempt=attempt)
            started = time.perf_counter()
            try:
                response = self._send_with_redirects(request, cancellation=cancellation)
            except (socket.timeout, TimeoutError) as exc:
                last_error = ApiTimeoutError("The upstream API request timed out.")
            except ApiCancelledError:
                raise
            except (BlockedHostError, SsrfPolicyViolationError, ResponseTooLargeError):
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_error = UpstreamApiError(
                    "The upstream API request failed before a response was received.",
                    details={"reason": type(exc).__name__},
                )
            else:
                status = response.status
                if status == 429:
                    last_error = ApiRateLimitError(
                        "The upstream API rate limit was reached.",
                        details={"status_code": status, "retry_after": response.headers.get("retry-after", "")},
                    )
                    self._emit("api.call.rate_limited", request, status_code=status, attempt=attempt)
                elif status in request.retry_statuses and attempt < attempts and self._retry_allowed(request):
                    last_error = UpstreamApiError(
                        f"The upstream API returned retryable status {status}.",
                        details={"status_code": status},
                    )
                else:
                    latency = (time.perf_counter() - started) * 1000
                    result = self._result(request, response, latency_ms=latency, attempts=attempt)
                    self._emit(
                        "api.call.completed" if result.upstream_ok else "api.call.failed",
                        request,
                        status_code=status,
                        latency_ms=round(latency, 3),
                        attempt=attempt,
                        error_code="" if result.upstream_ok else "upstream_api_failure",
                    )
                    return result
            if attempt < attempts and self._retry_allowed(request):
                delay = min(
                    30.0,
                    request.retry_backoff_seconds * (2 ** (attempt - 1)),
                )
                self._emit("api.call.retry", request, attempt=attempt, delay_seconds=delay)
                self.sleep(delay)
                continue
            break
        self._emit("api.call.failed", request, error_code=getattr(last_error, "code", "upstream_api_failure"))
        if last_error:
            raise last_error
        raise UpstreamApiError("The upstream API request failed.")

    def fetch_documentation(self, url: str) -> tuple[bytes, str]:
        """Policy-controlled documentation fetcher used by DocumentationImporter."""
        request = BuiltApiRequest(
            integration_id="api_documentation_fetch",
            operation_id="documentation_fetch",
            method="GET",
            url=url,
            headers={"Accept": "application/json, application/yaml, text/yaml, text/html, text/plain"},
            timeout_seconds=30,
            risk_level=OperationRiskLevel.READ_ONLY,
        )
        response = self._send_with_redirects(
            request,
            cancellation=None,
            stop_on_authorization_redirect=True,
        )
        if response.status >= 400:
            raise UpstreamApiError(
                f"Documentation URL returned HTTP {response.status}.",
                details={"status_code": response.status},
            )
        return response.body, response.headers.get("content-type", "application/octet-stream")

    def fetch(self, url: str) -> tuple[bytes, str]:
        """Satisfy the policy-controlled DocumentationFetcher contract."""
        return self.fetch_documentation(url)

    def _send_with_redirects(
        self,
        request: BuiltApiRequest,
        *,
        cancellation: threading.Event | None,
        stop_on_authorization_redirect: bool = False,
    ) -> "_RawResponse":
        url = request.url
        redirects: list[str] = []
        method = request.method
        body = request.body
        current_headers = dict(request.headers)
        for redirect_count in range(self.network_policy.maximum_redirects + 1):
            self._check_cancelled(cancellation)
            policy = self.network_policy
            parsed = urlsplit(url)
            approved_host = request.approved_network_host.rstrip(".").lower()
            if approved_host and (parsed.hostname or "").rstrip(".").lower() == approved_host:
                policy = policy.model_copy(
                    update={
                        "allowed_hosts": tuple(
                            sorted({*policy.allowed_hosts, approved_host})
                        ),
                        "allow_http": bool(
                            policy.allow_http or request.allow_insecure_http_once
                        ),
                    }
                )
            resolved = validate_network_target(url, policy)
            response = self.transport.send(
                method,
                url,
                resolved_ip=resolved,
                headers=current_headers,
                body=body,
                connect_timeout=request.connect_timeout_seconds,
                read_timeout=request.read_timeout_seconds,
                maximum_bytes=self.network_policy.maximum_response_bytes,
                cancellation=cancellation,
            )
            response.redirects = tuple(redirects)
            if response.status not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            next_url = _normalized_redirect_target(url, location)
            if stop_on_authorization_redirect and _is_oauth_authorization_redirect(
                next_url
            ):
                validate_network_target(next_url, self.network_policy)
                raise DocumentationAuthorizationRequiredError(
                    "The documentation source redirected to an OAuth authorization portal.",
                    details={
                        "authorization_origin": _origin_text(next_url),
                        "rendered_browser_inspection_available": True,
                    },
                )
            if redirect_count >= self.network_policy.maximum_redirects:
                raise UpstreamApiError("The upstream API exceeded the redirect limit.")
            # The next target is independently resolved and checked on the next loop.
            redirects.append(
                redact_mapping(next_url, secret_values=request.secret_values)
            )
            if _origin(next_url) != _origin(url):
                current_headers = {
                    key: value
                    for key, value in current_headers.items()
                    if key.lower()
                    not in {"authorization", "proxy-authorization", "cookie"}
                }
            if response.status == 303:
                method, body = "GET", None
            elif response.status in {301, 302} and method == "POST":
                method, body = "GET", None
            url = next_url
        raise UpstreamApiError("The upstream API exceeded the redirect limit.")

    def _result(
        self,
        request: BuiltApiRequest,
        response: "_RawResponse",
        *,
        latency_ms: float,
        attempts: int,
    ) -> ApiExecutionResult:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        json_body: Any = None
        text_body = ""
        file_reference = ""
        binary_metadata: dict[str, Any] = {}
        if content_type == "application/json" or content_type.endswith("+json"):
            try:
                json_body = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                text_body = response.body.decode("utf-8", errors="replace")
                content_type = "text/plain"
        elif content_type.startswith("text/") or content_type in {
            "application/xml",
            "application/javascript",
            "application/x-yaml",
        }:
            text_body = response.body.decode("utf-8", errors="replace")
        else:
            digest = hashlib.sha256(response.body).hexdigest()
            binary_metadata = {
                "bytes": len(response.body),
                "sha256": digest,
                "filename": _response_filename(response.headers),
            }
            if self.artifact_directory is not None:
                self.artifact_directory.mkdir(parents=True, exist_ok=True)
                try:
                    self.artifact_directory.chmod(0o700)
                except OSError:
                    pass
                filename = binary_metadata["filename"] or f"api-{digest[:16]}{mimetypes.guess_extension(content_type) or '.bin'}"
                target = self.artifact_directory / f"{digest[:16]}-{Path(str(filename)).name}"
                target.write_bytes(response.body)
                try:
                    target.chmod(0o600)
                except OSError:
                    pass
                file_reference = str(target)
        body_kind = "json" if json_body is not None else "text" if text_body else "file" if file_reference else "binary_metadata"
        rate_limit = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
        }
        safe_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"set-cookie", "authorization", "proxy-authorization"}
        }
        return ApiExecutionResult(
            integration_id=request.integration_id,
            operation_id=request.operation_id,
            method=request.method,
            redacted_url=redact_mapping(
                redact_url(request.url),
                secret_values=request.secret_values,
            ),
            status_code=response.status,
            headers=redact_mapping(safe_headers, secret_values=request.secret_values),
            content_type=content_type,
            body_kind=body_kind,
            json_body=redact_mapping(json_body, secret_values=request.secret_values),
            text_body=redact_mapping(text_body, secret_values=request.secret_values),
            file_reference=file_reference,
            binary_metadata=binary_metadata,
            latency_ms=latency_ms,
            redirects=response.redirects,
            rate_limit=rate_limit,
            attempts=attempts,
            upstream_ok=200 <= response.status < 400,
        )

    @staticmethod
    def _retry_allowed(request: BuiltApiRequest) -> bool:
        return request.method in SAFE_METHODS or request.retry_non_idempotent

    @staticmethod
    def _check_cancelled(cancellation: threading.Event | None) -> None:
        if cancellation and cancellation.is_set():
            raise ApiCancelledError("The API request was cancelled.")

    def _emit(self, kind: str, request: BuiltApiRequest, **payload: Any) -> None:
        if self.event_sink:
            parsed = urlsplit(request.url)
            self.event_sink(
                kind,
                {
                    "integration_id": request.integration_id,
                    "operation_id": request.operation_id,
                    "method": request.method,
                    "redacted_host_path": f"{parsed.hostname or ''}{parsed.path}",
                    **redact_mapping(payload, secret_values=request.secret_values),
                },
            )


@dataclass
class _RawResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    redirects: tuple[str, ...] = ()


def validate_network_target(url: str, policy: NetworkAccessPolicy) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in policy.allowed_schemes:
        raise SsrfPolicyViolationError("Only policy-allowed HTTP(S) URL schemes may be called.")
    if parsed.scheme == "http" and not policy.allow_http:
        raise SsrfPolicyViolationError("Plain HTTP is disabled by network policy.")
    if parsed.username or parsed.password:
        raise SsrfPolicyViolationError("URLs containing credentials are forbidden.")
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise BlockedHostError("The API URL does not contain a hostname.")
    if policy.allowed_hosts and hostname not in {item.lower() for item in policy.allowed_hosts}:
        raise BlockedHostError("The API hostname is not on the configured allowlist.")
    trusted_host = hostname in {item.lower() for item in policy.trusted_internal_hosts}
    trusted_networks = tuple(ipaddress.ip_network(item) for item in policy.trusted_internal_networks)
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise BlockedHostError("The API hostname could not be resolved.") from exc
    if not addresses:
        raise BlockedHostError("The API hostname did not resolve to an address.")
    validated: list[str] = []
    for raw in sorted(addresses):
        address = ipaddress.ip_address(raw)
        trusted_address = any(address in network for network in trusted_networks)
        if not address.is_global and not trusted_host and not trusted_address:
            raise SsrfPolicyViolationError(
                "The API hostname resolves to a loopback, private, link-local, multicast, reserved, "
                "or otherwise non-global address.",
                details={"hostname": hostname},
            )
        validated.append(raw)
    # Deterministic selection pins the connection to an already-validated result.
    return validated[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        *,
        pinned_ip: str,
        port: int,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        super().__init__(host, port=port, timeout=connect_timeout)
        self._pinned_ip = pinned_ip
        self._read_timeout = read_timeout

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock.settimeout(self._read_timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        pinned_ip: str,
        port: int,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=connect_timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_ip = pinned_ip
        self._read_timeout = read_timeout

    def connect(self) -> None:
        raw_socket = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)
        self.sock.settimeout(self._read_timeout)


def _pinned_request(
    method: str,
    url: str,
    *,
    resolved_ip: str,
    headers: dict[str, str],
    body: bytes | None,
    connect_timeout: float,
    read_timeout: float,
    maximum_bytes: int,
    cancellation: threading.Event | None,
) -> _RawResponse:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_class = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    connection = connection_class(
        host,
        pinned_ip=resolved_ip,
        port=port,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    try:
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        chunks: list[bytes] = []
        total = 0
        while True:
            if cancellation and cancellation.is_set():
                raise ApiCancelledError("The API request was cancelled.")
            chunk = response.read(min(64 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ResponseTooLargeError(
                    f"The API response exceeded the {maximum_bytes}-byte limit."
                )
            chunks.append(chunk)
        return _RawResponse(
            status=int(response.status),
            headers={str(key).lower(): str(value) for key, value in response.getheaders()},
            body=b"".join(chunks),
        )
    finally:
        connection.close()


def _response_filename(headers: dict[str, str]) -> str:
    disposition = headers.get("content-disposition", "")
    marker = "filename="
    if marker not in disposition.lower():
        return ""
    value = disposition[disposition.lower().index(marker) + len(marker) :].split(";", 1)[0]
    return Path(value.strip().strip("\"'")).name


def _request_fingerprint(request: BuiltApiRequest) -> str:
    payload = {
        "integration_id": request.integration_id,
        "operation_id": request.operation_id,
        "method": request.method,
        "url": request.url,
        "headers": request.headers,
        "body_sha256": hashlib.sha256(request.body or b"").hexdigest(),
        "session_id": request.session_id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    return (
        parsed.scheme.lower(),
        (parsed.hostname or "").lower(),
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


def _normalized_redirect_target(base_url: str, location: str) -> str:
    """Encode spaces in Location while rejecting header control-byte injection."""
    if any(
        character != " "
        and (ord(character) < 0x20 or ord(character) == 0x7F)
        for character in location
    ):
        raise UpstreamApiError(
            "The upstream API returned a redirect containing forbidden control characters."
        )
    encoded = quote(location, safe="/:?#[]@!$&'()*+,;=%")
    return urljoin(base_url, encoded)


def _is_oauth_authorization_redirect(url: str) -> bool:
    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    return {"client_id", "redirect_uri"}.issubset(query) and bool(
        {"response_type", "grant_type", "scope"}.intersection(query)
    )


def _origin_text(url: str) -> str:
    scheme, host, port = _origin(url)
    default_port = 443 if scheme == "https" else 80
    return f"{scheme}://{host}" + (f":{port}" if port != default_port else "")
