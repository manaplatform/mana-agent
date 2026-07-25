"""Transport-only SSH failure classification used for safe failover."""

from __future__ import annotations

from enum import Enum


class TransportFailure(str, Enum):
    SANDBOX_RESTRICTION = "sandbox_restriction"
    DNS_FAILURE = "dns_failure"
    ROUTE_UNAVAILABLE = "route_unavailable"
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "timeout"
    HOST_KEY_FAILURE = "host_key_failure"
    AUTHENTICATION_FAILURE = "authentication_failure"
    REMOTE_COMMAND_FAILURE = "remote_command_failure"
    UNKNOWN = "unknown"


def classify_ssh_failure(stderr: str, exit_code: int | None = None) -> TransportFailure:
    text = stderr.casefold()
    if any(item in text for item in ("operation not permitted", "sandbox", "socket: permission denied", "network is unreachable: operation not permitted")):
        return TransportFailure.SANDBOX_RESTRICTION
    if "could not resolve hostname" in text or "name or service not known" in text:
        return TransportFailure.DNS_FAILURE
    if "no route to host" in text or "network is unreachable" in text:
        return TransportFailure.ROUTE_UNAVAILABLE
    if "connection refused" in text:
        return TransportFailure.CONNECTION_REFUSED
    if "connection timed out" in text or "operation timed out" in text:
        return TransportFailure.TIMEOUT
    if any(item in text for item in ("host key verification failed", "remote host identification has changed", "no matching host key type")):
        return TransportFailure.HOST_KEY_FAILURE
    if any(item in text for item in ("permission denied", "too many authentication failures", "publickey")):
        return TransportFailure.AUTHENTICATION_FAILURE
    if exit_code not in (None, 0):
        return TransportFailure.REMOTE_COMMAND_FAILURE
    return TransportFailure.UNKNOWN


def permits_external_worker_failover(failure: TransportFailure) -> bool:
    return failure is TransportFailure.SANDBOX_RESTRICTION
