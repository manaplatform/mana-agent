# Computer Control and Desktop Automation

Mana-Agent has an optional, provider-neutral computer-control integration for
local desktop and installed-application actions. It is **disabled by default**.
All clients use the same `ComputerControlService`; CLI chat, Textual, dashboard,
gateway, and remote connectors do not execute operating-system automation
directly.

## Security model

An action runs only after this sequence:

```text
authenticated client context
  → typed model action decision
  → registered operation/argument validation
  → client and remote policy
  → exact permission scope
  → operating-system capability/permission
  → exact-action confirmation when required
  → provider or application adapter
  → timeout/cancellation
  → sanitized live event and audit record
```

There is no general `run_computer_command` tool. Models cannot supply shell,
AppleScript, PowerShell, D-Bus, COM, JavaScript, or accessibility program text.
Production providers construct fixed argument arrays and pass content as
separate arguments or stdin. Unknown operations, arguments, risks, permission
scopes, platforms, or model decisions stop without a fallback action.

The browser-desktop tools never expose saved passwords, cookies, authentication
tokens, payment data, private browser storage, or private form fields.
Clipboard, note, calendar, page, and screenshot contents are returned only to
the selected action. Audit records and live events record that sensitive
content was accessed without duplicating it.

## Enable and configure

Use Settings → Computer control in the Textual configuration UI, or Dashboard →
Computer Control for the complete permission editor and live capability matrix.
The equivalent `~/.mana/config.toml` section is:

```toml
MANA_COMPUTER_CONTROL_ENABLED = true

[computer_control]
enabled = true
provider = "auto"
allowed_clients = ["local_cli", "tui", "dashboard"]
allow_remote_control = false
remote_sensitive_scopes = []
require_local_confirmation_for_high_risk = true
require_confirmation = true
audit_enabled = true
audit_retention_days = 30
timeout_seconds = 30
allowed_paths = ["/Users/me/Documents"]

[computer_control.permissions]
"computer.apps.read" = "ask"
"computer.apps.control" = "ask"
"computer.calendar.read" = "ask"
"computer.calendar.write" = "ask"
"computer.media.read" = "ask"
"computer.media.control" = "ask"
"computer.notes.read" = "ask"
"computer.notes.write" = "ask"
"computer.browser.tabs.read" = "ask"
"computer.browser.page.read" = "ask"
"computer.browser.control" = "ask"
"computer.clipboard.read" = "ask"
"computer.clipboard.write" = "ask"
"computer.files.read" = "ask"
"computer.files.write" = "ask"
"computer.screenshot.capture" = "ask"
"computer.notifications.send" = "ask"
"computer.system.read" = "ask"
"computer.system.control" = "ask"

[computer_control.defaults]
browser = "auto"
calendar = "auto"
music = "auto"
notes = "auto"
```

Persistent configuration values are `denied`, `ask`, and `always`. `allow_once`
and `allow_session` are ephemeral runtime grants issued by a trusted approval UI
and are never restored from configuration. Persistent decisions store only
scope names and decisions, never private desktop content.

When a scope is `ask`, Mana-Agent does not end with instructions to edit
configuration. It places the exact pending action inside the active chat.
Textual opens an in-chat permission modal, the Dashboard chat timeline renders
an actionable permission card, and Dashboard → Computer Control also shows it with
**Deny**, **Allow once**, **This session**, and **Always** choices. Approval
immediately executes the stored action; the model cannot alter or approve it.
The local CLI equivalent is
`/computer-permission <request-id> once|session|always`. Permission requests
expire after two minutes, and remote clients cannot approve them.

To revoke access, set the scope to `denied`, remove its entry from
`~/.mana/computer_control_permissions.json`, revoke Mana-Agent in the operating
system privacy settings, or set `enabled = false` to disable the integration.

## Risk and confirmation

- Low: application open, media pause, current-track/system-status reads, URL open.
- Medium: private note/page/clipboard reads, calendar writes, file moves,
  notifications, application close, screenshots.
- High: delete a note/event, Trash a file, lock the computer.
- Critical: sleep, restart, shutdown, sign out, permanent deletion, publication,
  purchases, agreements, or external communication.

High and critical actions require a short-lived, single-use token bound to the
complete action—including execution ID, target, operation, arguments, risk,
permission, and source model decision. A token cannot authorize a changed
target or a later action. Critical actions are never silently chained.
The tool initially returns only a pending request ID; it is unusable until the
user presses **Approve exact action** in Dashboard → Computer Control or runs
`/computer-confirm <request-id>` in a trusted local CLI/Textual client. Approval
executes the stored exact action directly; it does not ask the model to recreate
or alter it. The model cannot approve its own request.

## Remote clients

Remote control is off even when local computer control is enabled. Enabling
`allow_remote_control` is not enough to expose personal data:
`remote_sensitive_scopes` must separately list each calendar, notes, browser,
clipboard, or screenshot read scope. Telegram, A2A, and API sessions retain
their authenticated gateway frontend identity and never inherit local-client
permissions. High/critical remote actions require trusted local confirmation by
default.

## Capability matrix

Capability discovery checks the current platform, graphical session, installed
utilities, and discovered applications. A group may be partially implemented;
an unsupported operation still returns a typed unavailable error.

| Provider | Implemented safe controls |
| --- | --- |
| macOS native | Bundle-ID application discovery/open, default-browser URL open, clipboard, Apple Music transport, system volume, notifications, full-screen screenshots, allowed-path open/reveal/copy/move/rename/mkdir/Trash/metadata. |
| Windows native | Shell application open, default-browser URL open, clipboard, allowed-path open/reveal/copy/move/rename/mkdir/Recycle Bin/metadata. |
| Linux freedesktop | `.desktop` discovery/launch, default-browser URL open, MPRIS transport via `playerctl`, Wayland/X11 clipboard when supported, notifications, PulseAudio volume, allowed-path open/reveal/copy/move/rename/mkdir/Trash/metadata. |
| Fake | All capability groups for tests only; never controls the developer or CI desktop. |

Calendar, notes, desktop-browser page inspection, active-window/selected-display
capture, Windows media/system controls, and Linux portal screenshots are
reported unavailable until a native adapter is installed. This is intentional:
Mana-Agent does not claim support or fall back to brittle coordinates when a
secure provider does not exist.

### Operating-system permissions

- macOS may require Automation/Apple Events, Accessibility, and Screen Recording
  access under System Settings → Privacy & Security.
- Windows may require an interactive desktop and the relevant application/COM or
  WinRT access. Recycle Bin operations use the native FileIO behavior.
- Linux requires a graphical session. Wayland features depend on desktop portals
  or Wayland-native tools; X11-only tools are not assumed. `gio`, `gtk-launch`,
  `playerctl`, `wl-clipboard`/`xclip`, `notify-send`, and `pactl` are discovered
  independently.
- Headless sessions report desktop capabilities unavailable. Filesystem
  operations remain bounded by `allowed_paths`.

## Tools

The route exposes narrow `computer_*`, `calendar_*`, `media_*`, `notes_*`,
desktop `browser_*`, and `clipboard_*` tools. Each mutating/sensitive tool
description states the accessed data, mutation, permission, and confirmation
requirement. File tools use only configured roots and `computer_trash_path`
always selects Trash/Recycle Bin; permanent deletion is not exposed.

Desktop browser tools are distinct from Mana-Agent's isolated Playwright browser
connector. The entry-routing model selects the `computer` source for installed
desktop application control and the `browser` source for isolated public-page
inspection.

## Events, cancellation, and audit

The service emits capability discovery, permission check, waiting permission,
waiting confirmation,
adapter selected, action start/completion/denial/cancellation/failure events into
the existing chat event stream. The same stream reaches Textual and dashboard
clients. `/cancel` cancels the active provider process and prevents later
workflow actions. Every action has a validated timeout.

Sanitized audit JSONL is stored at
`~/.mana/audit/computer-control.jsonl` with owner-only permissions and configured
retention. It contains IDs, client, capability/operation, target application,
risk, permission/confirmation outcomes, timestamps, final state, and sanitized
error code—never private content or screenshot pixels.

## Extending providers and adapters

1. Implement `ComputerControlProvider` in a platform package.
2. Return only capabilities whose production operation exists and is tested.
3. Construct fixed native API/argv calls; never accept model program text.
4. Validate application IDs, URLs, resource IDs, and paths.
5. Add fake/mocked tests runnable on Linux CI and separate genuine integration
   tests with platform/application/permission skips.
6. Register the provider in `discovery.default_registry`.

Application-specific integrations implement `ApplicationAdapter`, declare
platforms and capability groups, and register with
`ApplicationAdapterRegistry`. Selection order is user preference, active
application, configured default, best available adapter, then the generic OS
provider. Unavailable adapters never make the whole integration fail.
