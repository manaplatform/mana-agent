from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status

from .service import GitHubAutopilotService
from .signatures import verify_signature

router = APIRouter(prefix="/integrations/github", tags=["github-app"])


def _service(request: Request) -> GitHubAutopilotService:
    service = getattr(request.app.state, "github_autopilot", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="GitHub Autopilot is unavailable")
    return service


@router.post("/webhooks", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(request: Request) -> dict[str, object]:
    service = _service(request)
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    delivery_id = str(request.headers.get("X-GitHub-Delivery") or "").strip()
    event_name = str(request.headers.get("X-GitHub-Event") or "").strip()
    if not verify_signature(raw_body, signature, service.settings.webhook_secret):
        service.metrics["signature.invalid"] += 1
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid GitHub webhook signature")
    if not delivery_id or not event_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing GitHub delivery or event header")
    if len(raw_body) > 10 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="GitHub webhook payload is too large")
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed GitHub webhook payload") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="GitHub webhook payload must be an object")
    receipt = await service.accept(delivery_id, event_name, payload)
    return {"accepted": receipt.accepted, "delivery_id": receipt.delivery_id, "result": receipt.result, "job_id": receipt.job_id, "reason": receipt.reason}


@router.get("/health")
async def github_health(request: Request) -> dict[str, object]:
    return _service(request).health()


@router.get("/ready")
async def github_ready(request: Request) -> dict[str, object]:
    report = _service(request).readiness()
    if not report["ready"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=report)
    return report


@router.get("/metrics")
async def github_metrics(request: Request) -> dict[str, object]:
    return _service(request).metrics_snapshot()
