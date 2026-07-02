# NestJS Skill

## When to use

Use this skill for NestJS modules, controllers, providers, guards, interceptors, decorators, and dependency injection wiring.

## Rules

- Register providers and controllers in the owning module instead of side-loading them elsewhere.
- Keep injectable services responsible for business logic and controllers thin.
- Preserve DTO validation and pipe behavior when changing request shapes.
- Update module imports/exports when a provider is shared across boundaries.
- Test controller routing and provider behavior with the project's existing test setup.

## Verification

Run the focused NestJS test or build command for the changed module when available.
