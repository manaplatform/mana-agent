# Installation

## Requirements

- Python 3.11 or newer is recommended.
- A virtual environment is strongly recommended for local development.

## Install from source

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the project in editable mode:

```bash
pip install -e .
```

Install the broad optional runtime on workstations or headless servers with:

```bash
pip install -e ".[full]"
```

On Linux, `full` deliberately excludes native global keyboard/pointer capture.
That capability depends on `pynput` and the source-built `evdev` bindings, which
require Linux input headers and are not needed for server management, the
dashboard, automations, protocols, browser tools, or semantic Teach Mode.

Install Linux desktop capture explicitly only on a graphical workstation:

```bash
sudo apt-get install build-essential python3-dev linux-libc-dev
pip install -e ".[teach-desktop]"
```

The system package names vary by distribution. A headless server should use
`.[full]` without `.[teach-desktop]`.

## Verify the installation

Run the project checks if available in your environment:

```bash
pytest
```

If the repository provides additional validation commands, run them as well to confirm the environment is ready.
