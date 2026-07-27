# Mana-Agent GitHub App

Fleet verification from GitHub remains subject to the same webhook delivery
persistence, actor authorization, repository allowlist, and draft-only repair
policy as other Autopilot actions. Untrusted issue, review, workflow, and log
text cannot provide raw remote commands or grant Fleet permissions. Fleet is
disabled by default.

Mana GitHub Autopilot is a webhook-driven GitHub App integration. It accepts supported repository events, persists each delivery before scheduling work, authorizes the relevant installation and actor, and sends validated coding tasks through `CodexCodingAgentShim`. It does not poll, schedule GitHub Actions, merge pull requests, or fall back to the legacy coding agent.

## GitHub App setup

Create a GitHub App with these repository permissions:

- Metadata: read
- Contents: read and write
- Issues: read and write
- Pull requests: read and write
- Actions: read
- Checks: read
- Dependabot alerts: read
- Code scanning alerts: read
- Secret scanning alerts: read

Grant Workflows write only when `MANA_GITHUB_WORKFLOW_FILES_WRITE_ENABLED=true` and repository policy permits Mana to edit `.github/workflows/*`; administration access is not required. Subscribe only to Issues, Issue comment, Workflow run, Pull request review, Pull request review comment, Dependabot alert, Code scanning alert, and Secret scanning alert events.

Set the webhook URL to `<PUBLIC_URL>/integrations/github/webhooks`. A callback URL is not required by the server; use `<PUBLIC_URL>/integrations/github/callback` only if the App settings demand a placeholder. Install the App only on repositories allowed by Mana policy.

Generate a private key in the GitHub App settings and store it in a host secret volume with owner-only permissions. Store the webhook secret in Mana's secure `secrets.toml` or inject it through the deployment secret manager. Never place either value in repository configuration.

Required configuration:

```text
MANA_GITHUB_AUTOPILOT_ENABLED=true
MANA_GITHUB_APP_ID=123456
MANA_GITHUB_APP_PRIVATE_KEY_PATH=/run/secrets/mana-github-app.pem
MANA_GITHUB_WEBHOOK_SECRET=<secure reference/value>
MANA_GITHUB_PUBLIC_WEBHOOK_URL=https://mana.example.com/integrations/github/webhooks
MANA_GITHUB_ALLOWED_REPOSITORIES=owner/repository
MANA_GITHUB_MINIMUM_ACTOR_PERMISSION=write
MANA_CODEX_ENABLED=true
```

Optional policy settings include `MANA_GITHUB_INVOCATION_NAME`, `MANA_GITHUB_FIX_LABEL`, `MANA_GITHUB_ALLOWED_ORGANIZATIONS`, `MANA_GITHUB_ALLOWED_WORKFLOWS`, `MANA_GITHUB_ALLOWED_BRANCHES`, `MANA_GITHUB_ACTOR_ALLOWLIST`, `MANA_GITHUB_SECURITY_EVENTS_ENABLED`, `MANA_GITHUB_ALLOW_BOTS`, worker/runtime/change limits, and `MANA_GITHUB_DRAFT_PR_ONLY`.

Actor allowlist entries may be GitHub logins or `team:organization/team-slug`. Team entries additionally require organization Members read permission so Mana can validate active membership; omit that permission when team allowlists are not used.

Run `mana-agent github-app doctor`, then `mana-agent github-app serve`. Deployment probes are `/integrations/github/health` and `/integrations/github/ready`.

## Event and security behavior

Supported executions are `issues.labeled` with `mana-fix`, explicit `@mana-agent` comments, failed `workflow_run.completed` events allowed by policy, requested-change reviews, explicitly invoked inline review comments, and enabled Dependabot/code-scanning/secret-scanning alerts. Unsupported events are durably recorded as ignored and never sent to a model.

Human-triggered tasks require the configured repository permission (write by default) and optional actor allowlist. GitHub-generated security events require explicit security-event policy. Webhook, repository, test, workflow, and alert content is untrusted. Secret scanning values are redacted at ingestion; source removal never claims to rotate or revoke a credential.

Each task uses a persistent identity such as `github:<installation>:<repository>:<subject-type>:<number>`, an isolated managed worktree, and a deterministic `mana/...` branch. Draft pull requests contain a stable session marker and remain drafts; Mana never merges automatically.
