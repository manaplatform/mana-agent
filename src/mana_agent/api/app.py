from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.routes.analyze import router as analyze_router
from mana_agent.api.routes.workspaces import router as workspaces_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mana-Agent API",
        version="0.0.13",
        description="HTTP API for Mana-Agent repository intelligence workflows.",
    )

    @app.exception_handler(ManaApiError)
    async def _mana_api_error_handler(_request: Request, exc: ManaApiError) -> JSONResponse:
        payload: dict[str, str] = {"detail": exc.detail}
        if exc.error:
            payload["error"] = exc.error
        return JSONResponse(status_code=exc.status_code, content=payload)

    app.include_router(analyze_router)
    app.include_router(workspaces_router)
    return app


app = create_app()
