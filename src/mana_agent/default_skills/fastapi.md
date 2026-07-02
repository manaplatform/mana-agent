# FastAPI Skill

## When to use

Use this skill for FastAPI apps, routers, dependencies, Pydantic schemas, ASGI entrypoints, and Uvicorn deployments.

## Rules

- Keep request validation in Pydantic models and dependency functions where the project already does so.
- Register routers through the existing application factory or main app module.
- Preserve async boundaries and avoid blocking I/O inside request handlers.
- Return explicit HTTP errors for invalid state instead of leaking internal exceptions.
- Cover endpoint behavior with focused tests or client-level smoke checks.

## Verification

Run focused API tests and import or compile the changed FastAPI modules.
