# Teach Mode

Teach Mode records one explicitly started human demonstration as redacted
semantic events, compiles those events into a typed Mana Flow, and replays the
flow only through Mana's existing permission and confirmation boundaries.

> Do it once. Mana learns it forever.

## First recording

```bash
mana-agent teach doctor
mana-agent teach start "Export my weekly report"
mana-agent teach explain "This date changes every week"
mana-agent teach status
mana-agent teach stop
```

For native desktop monitoring, install the optional adapter, record local
consent, open the OS privacy panes, and start a new desktop-enabled session:

```bash
pip install "mana-agent[teach-desktop]"
mana-agent teach grant --scope full --allow --open-settings
mana-agent teach doctor
mana-agent teach start "Export my weekly report" --desktop
```

On macOS, approve the exact terminal/Python/Mana-Agent executable under
**Privacy & Security → Accessibility** and **Input Monitoring**. Mana stores its
own grants separately under `~/.mana/teach/grants.json`; it cannot and does not
edit the operating system privacy database. Local consent can be revoked with:

```bash
mana-agent teach grant --scope full --revoke
```

`--desktop` launches a persisted background recorder bound to the session, so
capture continues after the start command exits. Pause, resume, stop, and cancel
control that process through the persisted session. The recorder captures:

- active application and window changes;
- focused accessibility role, label/title, and native identifier when exposed;
- keyboard shortcuts and navigation keys;
- printable typing as a redacted character count and review placeholder;
- pointer buttons and normalized screen-relative fallback position.

It never stores printable keys as a raw keylog. Text reconstruction must come
from a permitted semantic accessibility value-change source or be supplied
during review. Sensitive/redacted typing steps are always marked for review.
Continuous screenshots, clipboard contents, passwords, cookies, and tokens are
not captured.

After all local grants are recorded, ordinary `teach start "…"` also selects
desktop capture automatically. If an OS permission is missing, startup fails
with the exact required scope instead of silently creating an empty semantic-only
recording. Use `--no-desktop` only when you deliberately want semantic-only
capture.

Recording is local, visible in CLI/event surfaces, and limited to enabled event
sources. It does not capture continuous video, screenshots, clipboard contents,
or raw password text. Browser and computer-control integrations may submit
versioned semantic events. `teach doctor` reports native accessibility,
Playwright, voice, headless, and platform limitations without claiming an
unavailable capability works.

Interrupted `recording` and `paused` sessions are stored under
`~/.mana/teach/sessions` and can be resumed. Raw and normalized events are kept
separately under `~/.mana/teach/recordings`. Files use owner-only permissions
where supported and atomic replacement for session and flow documents.

## Review and inputs

`teach stop` creates a draft; it does not activate it. Review the compiled
steps, provenance, confidence, inferred inputs, permissions, and verification:

```bash
mana-agent teach review export-my-weekly-report
mana-agent teach edit export-my-weekly-report
```

Parameter inference is conservative and uses the demonstration plus explicit
explanations. Credentials are never normal inputs: secret inputs have no
plaintext default, are excluded from sharing, and must resolve through a
`secret://` reference. Manually edited YAML is schema-validated when loaded.

## Verification and replay

```bash
mana-agent teach replay export-my-weekly-report --mode dry_run \
  --input date=2026-07-31
mana-agent teach replay export-my-weekly-report --mode guided \
  --input date=2026-07-31
mana-agent teach replay export-my-weekly-report --mode normal \
  --input date=2026-07-31
```

Dry run resolves inputs and previews steps without external side effects. It
always returns `unverified`. Guided replay pauses on low-confidence and
sensitive steps. Normal replay requires configured execution adapters,
permissions, and confirmations. Completion is not success: a real replay is
`verified` only when every required observable final rule passes. Unsupported
verification rules fail closed.

Sending, deleting, purchasing, publishing, paying, posting, and submitting
retain explicit confirmation even when demonstrated. A recording never grants
permanent replay permission.

## Correcting a selector

When a semantic selector fails, capture the repeated action and update only the
failed step:

```bash
mana-agent teach repair export-my-weekly-report step-003 \
  --type playwright_role --value '{"role":"button","name":"Export"}'
```

The repaired flow becomes a new revision. Previous selectors remain ranked
fallback candidates with failure history; unaffected steps are preserved.
Replay should continue from a safe application checkpoint once the selected
execution adapter supports checkpoint continuation.

## Automation handoff

Only reviewed flows with a successful verified replay can become automations.
Ask through chat; the handoff exposes safe flow metadata and pins the exact
version:

```bash
mana-agent chat
# You: Run the reviewed export-my-weekly-report flow every Friday at 4 PM.
```

The automation stores `flow_id`, `flow_version`, validated inputs, permission
scope references, and verification metadata. It never copies raw recording
events, selectors, desktop grants, or captured keyboard/mouse data. Updating
the pinned version is an explicit automation update. Background execution still
enforces permissions, confirmation, and final verification; a headless run
cannot confirm a sensitive action.

## Export and import

```bash
mana-agent teach export export-my-weekly-report \
  --output weekly-report.mana-flow
mana-agent teach import weekly-report.mana-flow
mana-agent teach replay export-my-weekly-report --mode dry_run
```

The deterministic archive contains `manifest.yaml`, `flow.yaml`,
`permissions.yaml`, `selectors.json`, `verification.yaml`, and `README.md`.
Export scans for tokens, keys, cookies, card-like values, email/account text,
and machine-specific home paths and blocks unsafe packages. There is no silent
override.

Imports are untrusted: exact structure, paths, links, sizes, checksum, schemas,
and sensitive content are validated. Imported flows remain
`imported_pending`; only dry run is allowed until review and explicit
activation. Packages contain declarative data and no embedded executable code.

## Flow Cards and privacy

```bash
mana-agent teach card export-my-weekly-report --minutes-saved 20
```

Cards are private local metadata derived from recording duration, application
and action counts, and verified replay history. Mana never publishes a card or
replay automatically. The interface reserves an export-ready representation
for a future gallery but has no hosted service.

## Configuration

The `[teach]` table in `~/.mana/config.toml` supports:

```toml
[teach]
enabled = true
event_sources = ["browser", "accessibility", "application", "filesystem"]
retention_days = 30
screenshot_policy = "never"
coordinate_fallback = true
voice_enabled = false
browser_capture = true
excluded_applications = []
allowed_applications = []
excluded_domains = []
recording_allowed_paths = []
sensitive_detection = true
automatic_verification = true
replay_retry_limit = 1
correction_checkpoints = true
flow_cards = true
experimental_sharing = false
desktop_capture = false
```

Secure defaults disable screenshot and voice persistence, enable redaction,
keep sharing private, and require imported flows to dry-run.

## Platform support

| Capability | macOS | Windows | Linux/headless |
| --- | --- | --- | --- |
| Semantic events from Mana tools | Yes | Yes | Yes |
| Browser DOM events | With Playwright extra | With Playwright extra | With Playwright extra |
| Native accessibility | ApplicationServices adapter | Optional UI Automation adapter | Optional AT-SPI adapter |
| Keyboard/pointer monitor | `pynput` + OS grants | `pynput` + OS grants | `pynput` + desktop session |
| Active app/window | AppKit/ApplicationServices | Win32 foreground window | `xdotool` when installed |
| Coordinate fallback contract | AppKit screen-relative | Win32 screen-relative | Tk/display-relative |
| Voice | Optional adapter; off by default | Optional adapter; off by default | Optional adapter; off by default |

Native adapter imports are delayed, so missing OS packages never break core CLI
or tests. macOS has the complete native path. Windows and Linux provide global
keyboard/pointer capture plus active-window metadata where the listed local
facilities exist; richer platform accessibility event streams and voice
transcription remain extension work.

## Safe demonstration

A safe first flow is: open a public page, navigate to another public page, and
verify the final URL. It requires no private account and does not send, publish,
purchase, or delete anything.

## Troubleshooting

- Run `mana-agent teach doctor --json` for machine-readable capability state.
- If no active recording is found, pass `--session teach_...` for a recovered
  session.
- If replay is `unverified`, add an observable final rule instead of treating
  step completion as success.
- If export is blocked, replace every reported sensitive or machine-specific
  value; no unsafe-export flag exists.
- If an imported flow will not run normally, dry-run it, map inputs and
  applications, inspect permissions, then explicitly accept it.

## Architecture and extension points

`mana_agent.teach` separates typed models, contracts, storage, recording,
normalization, parameterization, compilation, replay, verification, correction,
redaction, packaging, platform diagnostics, application service, CLI, API, and
model tools. Protocols cover event recorders, accessibility/browser sources,
normalizers, compilers, executors, verifiers, selector repair, sensitive-data
detection, storage, and packaging. Progress uses the shared execution event hub,
so TUI, dashboard WebSockets, and gateway consumers receive `teach.*` events.

Future gallery work should consume privacy-previewed Flow Card and package
interfaces. It must not add automatic publication or weaken local validation.
