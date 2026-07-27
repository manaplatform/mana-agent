# Telegram connector

Mana-Agent can expose its existing model-routed chat service through a Telegram bot. Polling is the default and is appropriate for local machines and private servers. Webhooks reuse the Mana-Agent FastAPI server and require a public HTTPS endpoint.

## Create and configure a bot

1. In Telegram, open the verified `@BotFather` account, run `/newbot`, and retain the bot token as a deployment secret.
2. Run setup. The token is used once for `getMe` validation and is not written to TOML:

   ```bash
   mana-agent connector telegram setup \
     --repository /srv/projects/my-repo \
     --allowed-user 123456789
   export TELEGRAM_BOT_TOKEN='…'
   ```

3. Start polling:

   ```bash
   mana-agent connector telegram start
   ```

Use `/id` in a private chat or approved group to inspect the numeric user, chat, and forum-topic IDs. Authorization uses numeric IDs; usernames are never an authorization credential. By default, access is closed and groups are disabled.

## Configuration

Mana-Agent reads the existing `~/.mana/config.toml`. Secret values remain in environment variables named by `bot_token_env` and `secret_env`.

```toml
[telegram]
enabled = true
transport = "auto"
bot_token_env = "TELEGRAM_BOT_TOKEN"
allowed_users = [123456789]
allowed_chats = []
admin_users = [123456789]
open_access = false
groups_enabled = false
group_activation = "mention"
parse_mode = "MarkdownV2"
request_timeout_seconds = 30
max_message_length = 4096
default_repository = "/srv/projects/my-repo"
allowed_repository_roots = ["/srv/projects"]

[telegram.polling]
timeout_seconds = 30
drop_pending_updates = false
reconnect_max_seconds = 60

[telegram.webhook]
public_url = "https://agent.example.com"
path = "/integrations/telegram/webhook"
secret_env = "TELEGRAM_WEBHOOK_SECRET"
listen_host = "127.0.0.1"
listen_port = 8787
drop_pending_updates = false

[telegram.queue]
backend = "local"
max_attempts = 5
retry_delay_seconds = 2
concurrency = 4

[telegram.attachments]
enabled = false
max_bytes = 10485760
allowed_mime_types = ["text/plain", "application/pdf", "text/csv"]
```

Supported environment overrides are `MANA_TELEGRAM_ENABLED`, `MANA_TELEGRAM_TRANSPORT`, and `TELEGRAM_WEBHOOK_URL`. The token and webhook secret are resolved from the configured environment-variable names. `auto` selects webhook only when a public URL is configured; otherwise it selects polling. With no Telegram configuration, the connector remains disabled and all existing commands behave unchanged.

## Polling deployments

Polling calls `deleteWebhook` before `getUpdates`, uses Telegram long polling, and stores each update in SQLite before committing its offset. A per-token process lock prevents concurrent pollers. The collector and model workers are separate, so a long-running task does not block incoming updates.

`mana-agent connector telegram start` and `/connect telegram start` launch the
registered connector worker through Mana-Agent's persistent background-process
manager. The worker survives CLI, Textual, dashboard, and API frontend shutdown;
use `/processes` or either Telegram stop command for identity-checked graceful
shutdown and bounded forced cleanup.

## Webhook deployments

Set a random secret of at least 32 characters and a public HTTPS URL:

```bash
export TELEGRAM_BOT_TOKEN='…'
export TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"
export TELEGRAM_WEBHOOK_URL='https://agent.example.com'
mana-agent connector telegram webhook set
mana-agent connector telegram start
```

The endpoint validates `X-Telegram-Bot-Api-Secret-Token`, JSON content type, and request size. It durably queues valid updates and returns HTTP 200 without waiting for Mana-Agent execution. Duplicate `update_id` values do not execute twice.

Nginx example:

```nginx
location /integrations/telegram/webhook {
    proxy_pass http://127.0.0.1:8787;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto https;
}
```

Caddy example:

```caddy
agent.example.com {
    reverse_proxy 127.0.0.1:8787
}
```

Telegram does not permit `getUpdates` while a webhook is active. Polling startup removes the webhook; use `webhook set` when switching back. `webhook delete` removes it explicitly.

## Commands and groups

Telegram delegates `/help`, `/status`, `/new`, `/sessions`, `/tasks`, `/models`,
`/processes`, `/connect`, `/cancel`, and other supported commands to the shared
chat-command registry after access authorization. `/id` is a Telegram-only
shared definition. `/new` destructively replaces only the canonical session
bound to the current bot/chat/topic/sender context and persists the new mapping.

Group activation can require an explicit bot mention, a reply to the bot, any bot command, or `always`. Prefer mention or reply activation. Forum topics have independent ordered execution lanes and sessions. Independent chats can execute concurrently up to the configured queue concurrency.

## Approved Mutation Plan: Telegram messaging

The Telegram connector can be used as a safe delivery channel for approved `MutationPlan`s.

### Executable `MutationCommand` for an approved `MutationPlan`

For the approved plan `mp_e0dac3f34151`, the executable mutation contract form is:

```text
MutationCommand(mp_e0dac3f34151)
```

Run it deterministically via:

```bash
mana-agent run --root-dir /path/to/project --plan-id mp_e0dac3f34151
```

## Attachments and security

When enabled, attachments are checked by declared size and MIME type, downloaded through Telegram's file API, given a sanitized filename, parsed through Mana-Agent's document service in an isolated session directory, and removed after the turn. Files are never executed. Repository access remains limited to `default_repository` inside `allowed_repository_roots`; Telegram does not provide an unrestricted path selector.

Long responses are escaped for the configured parse mode and split below Telegram's 4096-character limit. If Telegram rejects formatting, the connector retries that response as plain text. Tokens, secrets, full messages, internal prompts, chain-of-thought, and unrestricted tool arguments are excluded from connector logs and status output.

`/connect telegram` uses a frontend secret field, validates `getMe` before
saving configuration, and stores only a keyring reference or explicit
environment-variable name. Raw tokens are excluded from history, events,
process records, command lines, exception rendering, and dashboard state.

## Operations and troubleshooting

```bash
mana-agent connector telegram --help
mana-agent connector telegram status
mana-agent connector telegram test
mana-agent connector telegram info
mana-agent connector telegram webhook set
mana-agent connector telegram webhook delete
```

- `401`: the token is invalid or revoked; obtain a current BotFather token.
- `403`: preserve Telegram's diagnostic—common cases include a user blocking the bot or the bot being removed from a group.
- `409`: another poller is active, or polling was attempted while a webhook remained registered.
- `429`: Telegram supplied `retry_after`; the client waits before retrying.

The dashboard Telegram page shows non-secret configuration and durable queue depth. Run `test` for live credential and webhook diagnostics. Accepted updates remain queued across restart; expired processing leases are recovered and exhausted updates enter the failed state.
