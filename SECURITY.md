# Security Policy

## Remote execution fabric

Task commands execute through a capability-checked sandbox provider. The local
provider explicitly reports that it cannot isolate networks; requests requiring
isolation fail closed. Routing decisions and persisted sandbox specifications
contain secret references only. Values are resolved at the provider boundary,
redacted from captured output where known, and excluded from events, snapshots,
artifacts, and errors. SSH host-key verification is enabled by default, artifact
paths are workspace-confined and symlink-safe, and cleanup failures never erase
the original task failure. See `docs/execution-providers.md` for enforcement
limitations that must be considered when enabling remote providers.

Mana-Agent takes security seriously. We appreciate responsible disclosure of security issues and will work to investigate, validate, and resolve legitimate vulnerabilities as quickly as possible.

This document describes how to report vulnerabilities, how reports are handled, and the security principles followed by Mana-Agent.

---

# Supported Versions

Security updates are provided for the latest stable release.

| Version | Supported |
|----------|-----------|
| Latest release | ✅ |
| Development branch (`main`) | Best effort |
| Older releases | ❌ |

If you discover a vulnerability affecting an unsupported version, please verify whether it also exists in the latest release before reporting it.

---

# Reporting a Vulnerability

Please report vulnerabilities privately.

**Primary contact**

- **Email:** root@manadev.net

Please include as much information as possible:

- Description of the vulnerability
- Expected security impact
- Steps to reproduce
- Proof of concept (if available)
- Affected versions
- Operating system
- Configuration details
- Relevant logs (with secrets removed)
- Screenshots if they help explain the issue

Please avoid publishing vulnerabilities publicly until we have had a reasonable opportunity to investigate and release a fix.

---

# Response Process

After receiving a report we will:

1. Acknowledge receipt.
2. Validate the report.
3. Assess severity and impact.
4. Develop and test a fix.
5. Coordinate disclosure when appropriate.
6. Publish a security release if necessary.

Response times may vary depending on complexity and available maintainer resources.

---

# Responsible Disclosure

We ask that researchers:

- Do not publicly disclose vulnerabilities before coordinated disclosure.
- Do not access data belonging to others.
- Do not intentionally disrupt systems or services.
- Do not exploit vulnerabilities beyond what is necessary to demonstrate the issue.
- Follow all applicable laws and regulations.

Good-faith security research is appreciated.

---

# Scope

Examples of issues that are generally considered security vulnerabilities include:

- Remote code execution
- Command injection
- Prompt injection leading to unauthorized actions
- Privilege escalation
- Authentication or authorization bypass
- Arbitrary file read/write outside approved boundaries
- Secret leakage
- Sandbox escape
- Path traversal
- SSRF
- Unsafe shell execution
- Credential exposure
- Supply-chain compromise
- Memory isolation failures
- Cross-agent privilege escalation

Issues such as feature requests, crashes without security impact, formatting bugs, or performance problems are generally not considered security vulnerabilities.

---

# Safe Handling of Sensitive Information

Never include:

- API keys
- OAuth tokens
- Session cookies
- Authorization headers
- SSH keys
- Private keys
- Passwords
- Secrets stored in environment variables

If sensitive information is required to reproduce an issue, share it privately through the reporting email.

---

# Email Connector Security

Email content is considered untrusted input.

This includes:

- Email bodies
- HTML
- Headers
- Attachments
- Embedded images
- Hyperlinks
- Calendar invites
- MIME metadata

These inputs **must never**:

- authorize actions
- override system instructions
- bypass confirmation requirements
- execute commands
- grant additional permissions

OAuth credentials are stored only in the operating system keychain/keyring and are never stored in:

- Mana-Agent configuration
- prompts
- logs
- repository artifacts
- generated reports

Sending email and destructive mailbox operations always require explicit user approval that is bound to the exact action payload.

---

# AI Security

Mana-Agent treats all model input as potentially adversarial.

Security principles include:

- Prompt injection resistance
- Tool permission boundaries
- Explicit approval for destructive actions
- Least-privilege tool execution
- Repository boundary enforcement
- Verification before applying changes
- Isolation between independent agents
- No automatic privilege escalation
- Human approval for sensitive operations

Agent outputs are never treated as trusted authority.

## Computer Control Security

Computer control is disabled by default, local-client-only by default, and
implemented through narrow typed tools. It never exposes a raw shell,
AppleScript, PowerShell, D-Bus, COM, JavaScript, accessibility-script, mouse
coordinate, or keyboard command input to the model. All action decisions,
arguments, application/resource identifiers, HTTP(S) URLs, and configured file
roots are validated before execution.

Calendar, notes, browser page, clipboard, and screenshot access have independent
scopes. An `ask` decision creates a short-lived, exact-action request in the
active trusted-local Textual/dashboard screen; remote clients cannot approve it,
and an allow choice resumes the stored action without a second model decision.
High and critical operations require a separate short-lived, single-use token
bound to the exact action and—when requested remotely—a trusted local
confirmation. Remote sessions retain their connector identity and cannot
inherit local permission. Unsupported and headless capabilities fail closed.

Audit and live events contain access metadata but never full note/page/clipboard
content, browser secrets/private storage, form values, tokens, cookies,
passwords, payment data, or screenshot pixels. File removal uses operating
system Trash/Recycle Bin; permanent deletion is not exposed. Full limitations
and revocation instructions are in
[`docs/22-computer-control.md`](docs/22-computer-control.md).

---

# Supply Chain Security

Mana-Agent attempts to reduce supply-chain risk by encouraging:

- Trusted package sources
- Dependency review
- Version pinning where appropriate
- Secure update practices
- Verification before executing generated code

Users remain responsible for reviewing generated code before execution.

---

# Secrets

Mana-Agent is designed to avoid exposing secrets.

Secrets should never appear in:

- logs
- telemetry
- prompts
- generated documentation
- commit messages
- issue reports

If a secret is accidentally exposed, rotate it immediately.

---

# Security Releases

Critical vulnerabilities may result in an out-of-band security release.

Release notes will include:

- affected versions
- fixed versions
- mitigation guidance
- upgrade recommendations

Sensitive implementation details may be withheld until users have had a reasonable opportunity to update.

---

# Security Best Practices

We recommend that users:

- Keep Mana-Agent updated.
- Review generated code before execution.
- Use least-privilege API credentials.
- Keep operating systems updated.
- Rotate credentials regularly.
- Avoid sharing secrets with language models.
- Verify third-party plugins before installation.

---

# Contact

Security questions or vulnerability reports:

**root@manadev.net**

Thank you for helping improve the security of Mana-Agent.
