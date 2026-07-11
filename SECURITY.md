# Security Policy

This project uses the Security guidance in `src/mana_agent/default_skills/security.md`.

## Reporting vulnerabilities

Report vulnerabilities to the location defined below. If this project later adds an explicit disclosure channel, this section should be updated.

- **Primary contact:** root@manadev.net
When reporting, please include:

- A description of the vulnerability and impact
- Steps to reproduce (if applicable)
- Affected versions / configuration
- Any logs or screenshots necessary to validate the issue (avoid including secrets)

## What we will do

- Acknowledge receipt of your report
- Assess severity and scope
- Coordinate a fix
- Publish a follow-up when appropriate

## Safe handling of sensitive data

- Do not include secrets (API keys, tokens, private keys, or authorization headers) in public reports.
- If you must provide sensitive material to reproduce an issue, share it privately using the contact above.

## Email connectors

Email bodies, headers, HTML, attachments, and links are untrusted external content and cannot authorize actions. OAuth credentials are stored in the OS keyring, never in Mana-Agent configuration, logs, prompts, or repository artifacts. Sending and destructive mailbox actions require approval bound to the exact payload.
