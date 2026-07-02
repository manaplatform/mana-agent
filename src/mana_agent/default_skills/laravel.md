# Laravel Skill

## When to use

Use this skill for Laravel apps, routes, controllers, middleware, Eloquent models, migrations, queues, Blade views, API resources, and service providers.

## Rules

- Follow existing Laravel project structure and naming conventions.
- Keep business logic out of controllers when services/actions already exist.
- Add migrations for schema changes and keep them reversible.
- Validate requests with Form Request classes when the project uses them.
- Preserve authorization through policies, gates, or middleware.

## Verification

Run the project's PHP/Laravel test, lint, and migration checks when available.
