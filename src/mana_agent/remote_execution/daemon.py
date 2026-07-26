"""Independent reverse-worker process using the validated local execution provider."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import platform
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization

from mana_agent import __version__
from mana_agent.remote_execution.credentials import CredentialStore, WorkerIdentity
from mana_agent.remote_execution.models import RemoteExecutionRequest, WorkerCapabilities, WorkerRegistration
from mana_agent.remote_execution.protocol import MessageType, WorkerMessage
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider
from mana_agent.fleet.capabilities import probe_worker_capabilities

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    coordinator_url: str
    worker_id: str
    name: str
    state_dir: Path
    heartbeat_interval_seconds: int = 15
    allow_insecure_http: bool = False
    allow_insecure_local_development: bool = False

    @property
    def websocket_url(self) -> str:
        parsed = urlparse(self.coordinator_url)
        if parsed.scheme == "https":
            return "wss" + self.coordinator_url[5:].rstrip("/") + "/api/v1/workers/connect"
        explicit_http = self.allow_insecure_http and parsed.scheme == "http"
        legacy_local_http = (
            self.allow_insecure_local_development
            and parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if explicit_http or legacy_local_http:
            return "ws" + self.coordinator_url[4:].rstrip("/") + "/api/v1/workers/connect"
        raise ValueError(
            "worker requires an HTTPS coordinator; HTTP requires explicit allow_insecure_http configuration"
        )


class ReverseWorkerDaemon:
    def __init__(self, config: WorkerRuntimeConfig, *, credentials: CredentialStore | None = None) -> None:
        self.config = config
        self.credentials = credentials or CredentialStore(config.state_dir)
        self.stop_event = asyncio.Event()
        self.running_jobs: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def registration(worker_id: str, name: str, public_key_pem: str) -> WorkerRegistration:
        return WorkerRegistration(worker_id=worker_id, display_name=name or socket.gethostname(),
                                  capabilities=WorkerCapabilities(), operating_system=platform.system(),
                                  ssh_available=False, architecture=platform.machine(), hostname=socket.gethostname(),
                                  mana_version=__version__, public_key_pem=public_key_pem)

    async def run(self) -> None:
        identity = self.credentials.load(self.config.worker_id)
        if identity is None:
            raise RuntimeError("worker identity is not enrolled; run `mana-agent worker install`")
        delay = 1.0
        while not self.stop_event.is_set():
            try:
                await self._connect(identity)
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("worker connection lost: %s", exc)
                await asyncio.sleep(min(60.0, delay) * random.uniform(0.8, 1.2))
                delay = min(60.0, delay * 2)

    def _signed(self, identity: WorkerIdentity, message: WorkerMessage) -> WorkerMessage:
        private = serialization.load_pem_private_key(identity.private_key_pem.encode(), password=None)
        message.signature = base64.b64encode(private.sign(message.signing_bytes())).decode()
        return message

    async def _connect(self, identity: WorkerIdentity) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise RuntimeError("reverse worker transport requires the mana-agent[workers] extra") from exc
        async with websockets.connect(self.config.websocket_url, max_size=1_048_576, open_timeout=15) as socket_client:
            hello = WorkerMessage(type=MessageType.HELLO, worker_id=identity.worker_id,
                                  payload={"credential": identity.credential, "name": self.config.name})
            await socket_client.send(hello.model_dump_json())
            authenticated = WorkerMessage.parse_frame(await socket_client.recv())
            if authenticated.type is not MessageType.AUTHENTICATED:
                raise PermissionError("coordinator did not authenticate worker")
            capabilities = await probe_worker_capabilities(
                identity.worker_id,
                labels=set(),
                max_concurrency=self.registration(
                    identity.worker_id, self.config.name, identity.public_key_pem
                ).max_concurrent_jobs,
                execution_providers={"reverse-worker"},
            )
            await socket_client.send(self._signed(
                identity,
                WorkerMessage(
                    type=MessageType.CAPABILITIES,
                    worker_id=identity.worker_id,
                    payload={"inventory": capabilities.model_dump(mode="json")},
                ),
            ).model_dump_json())
            heartbeat = asyncio.create_task(self._heartbeats(socket_client, identity))
            try:
                while not self.stop_event.is_set():
                    message = WorkerMessage.parse_frame(await socket_client.recv())
                    if message.type is MessageType.OFFER:
                        await self._handle_offer(socket_client, identity, message)
                    elif message.type is MessageType.CANCEL:
                        task = self.running_jobs.get(message.job_id)
                        if task:
                            task.cancel()
                    elif message.type is MessageType.SHUTDOWN:
                        self.stop_event.set()
            finally:
                heartbeat.cancel()
                for task in self.running_jobs.values():
                    task.cancel()

    async def _heartbeats(self, socket_client, identity: WorkerIdentity) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.config.heartbeat_interval_seconds)
            await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.HEARTBEAT,
                                                                            worker_id=identity.worker_id)).model_dump_json())

    async def _handle_offer(self, socket_client, identity: WorkerIdentity, message: WorkerMessage) -> None:
        if message.job_id in self.running_jobs:
            return
        try:
            request = RemoteExecutionRequest.model_validate(message.payload["request"])
            if request.worker_id != identity.worker_id or request.provider != "external_worker":
                raise ValueError("job is not assigned to this worker")
            if not request.read_only:
                # Existing coordinator permission state must be resolved before a
                # state-changing request is sent to this worker.
                raise PermissionError("worker only accepts approved read-only remote jobs")
        except Exception as exc:
            await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.REJECTED, worker_id=identity.worker_id,
                                                                            job_id=message.job_id, correlation_id=message.message_id,
                                                                            payload={"reason": str(exc)})).model_dump_json())
            return
        task = asyncio.create_task(self._execute(socket_client, identity, request))
        self.running_jobs[request.job_id] = task
        task.add_done_callback(lambda _: self.running_jobs.pop(request.job_id, None))

    async def _execute(self, socket_client, identity: WorkerIdentity, request: RemoteExecutionRequest) -> None:
        await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.ACCEPTED, worker_id=identity.worker_id,
                                                                        job_id=request.job_id)).model_dump_json())
        await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.STARTED, worker_id=identity.worker_id,
                                                                        job_id=request.job_id)).model_dump_json())
        try:
            cancel = asyncio.Event()

            def emit(event) -> None:
                kind = MessageType.STDOUT if event.kind == "stdout" else MessageType.STDERR
                if event.kind in {"stdout", "stderr"}:
                    outbound = WorkerMessage(type=kind, worker_id=identity.worker_id, job_id=request.job_id,
                                             payload={"chunk": str(event.data.get("chunk", ""))})
                    asyncio.create_task(socket_client.send(self._signed(identity, outbound).model_dump_json()))

            result = await LocalSSHProvider().execute(request, emit, cancel)
            code, _stdout, _stderr = result
            terminal = MessageType.COMPLETED if code == 0 else MessageType.FAILED
            await socket_client.send(self._signed(identity, WorkerMessage(type=terminal, worker_id=identity.worker_id,
                                                                            job_id=request.job_id,
                                                                            payload={"exit_code": code})).model_dump_json())
        except asyncio.CancelledError:
            await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.CANCELLED, worker_id=identity.worker_id,
                                                                            job_id=request.job_id)).model_dump_json())
            raise
        except Exception as exc:
            await socket_client.send(self._signed(identity, WorkerMessage(type=MessageType.FAILED, worker_id=identity.worker_id,
                                                                            job_id=request.job_id, payload={"error": str(exc)})).model_dump_json())


def write_worker_config(config: WorkerRuntimeConfig) -> Path:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.state_dir / "worker.json"
    config_path.write_text(json.dumps({"coordinator_url": config.coordinator_url, "worker_id": config.worker_id,
                                       "name": config.name, "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
                                       "allow_insecure_http": config.allow_insecure_http,
                                       "allow_insecure_local_development": config.allow_insecure_local_development}), encoding="utf-8")
    config_path.chmod(0o600)
    return config_path


def load_worker_config(state_dir: Path) -> WorkerRuntimeConfig:
    data = json.loads((state_dir / "worker.json").read_text(encoding="utf-8"))
    return WorkerRuntimeConfig(state_dir=state_dir, **data)
