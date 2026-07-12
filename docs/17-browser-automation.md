# Browser Automation

Mana-Agent can expose an isolated, headless Playwright browser directly to the
chat model. The model selects structured browser actions from the user's
natural-language request; Mana-Agent does not encode workflows for individual
websites.

## Installation

Playwright is optional. Install the browser extra and its managed Chromium
runtime before the first browser task:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
```

If Playwright or Chromium is unavailable, browser tools return an actionable
setup error while repository and chat features remain usable.

## Using the Browser from Chat

Start a normal chat session and describe the outcome you want:

```text
Check https://example.test and report the visible controls, page title, and any broken links.
```

For an interactive task, the routing model selects the browser workflow and the
browser operator follows this evidence-driven sequence:

1. `browser_open` opens the exact URL in an isolated session.
2. `browser_inspect` returns current controls, refs, URL, and page version.
3. The model chooses `browser_click`, `browser_type`, `browser_select`, or another
   browser action from that inspection.
4. The model inspects again after navigation or a changed page version.
5. CAPTCHA/MFA stops execution; sensitive final submission returns an exact
   confirmation challenge.

The terminal tool panel displays every browser tool actually started, completed,
or failed. Typed values are redacted from terminal output, traces, and memory.

```text
Create an account with the details I provide. Stop for CAPTCHA or MFA, and ask
before accepting terms or sending the final registration form.
```

```text
Complete this form and upload /absolute/path/to/resume.pdf. Save the downloaded
result, but ask before the final submission.
```

The model can open pages, inspect page text and interactive elements, click,
type, select options, scroll, wait, capture screenshots, upload files, save
downloads, navigate back, work with tabs and pop-ups, and close the session.
Tool results include the current URL and structured progress or error details.

## Sessions and State

- Each browser session uses an isolated browser context, including cookies and
  authentication state. Tabs, redirects, and pop-ups stay associated with that
  session.
- Sessions are ephemeral by default and are closed when explicitly requested
  or when their lifecycle ends.
- Authentication state is persisted only when explicitly enabled. Persisted
  state belongs under `${MANA_HOME:-~/.mana}/browser/`, outside the repository,
  with user-only filesystem permissions.
- Uploads must reference an allowed local file. Downloads are written only to
  the configured, isolated download directory and return the resolved path.
- Page content is untrusted external input. It cannot override Mana-Agent's
  tool permissions, safety policy, or model-decision contract.

## Confirmation and Security Boundaries

Mana-Agent pauses for exact-action user confirmation before irreversible or
sensitive actions, including payments, publishing content, deleting data,
accepting legal agreements, and sending a final form. If the target or submitted
content changes, the prior confirmation is no longer valid.

Interactive chat displays a one-time challenge. The user must enter
`/approve-browser <token>` and then ask Mana-Agent to continue. The model cannot
promote its own challenge; non-interactive execution stops at this boundary.

Mana-Agent never bypasses CAPTCHA, MFA, website security controls, access
restrictions, or authentication requirements. When one is encountered, the
browser stops safely and reports what the user must complete or authorize.

## Testing

Unit tests use deterministic fake browser objects. The optional Playwright
integration test uses a local HTTP server and never depends on a public website:

```bash
python -m pytest tests/connectors/test_browser_core.py -q
python -m pytest tests/integration/test_browser_playwright.py -q
```

The integration test is skipped with a clear reason when Playwright or its
Chromium runtime is not installed.

## Troubleshooting

- `playwright is not installed`: reinstall Mana-Agent's declared dependencies.
- `browser executable does not exist`: run `python -m playwright install chromium`.
- Upload rejected: use an existing absolute path permitted by the active
  workspace policy.
- Action requires confirmation: review the exact target and payload, then
  approve that action. Approval does not apply to a changed action.
- CAPTCHA or MFA encountered: complete the challenge yourself; Mana-Agent will
  not solve or circumvent it.
