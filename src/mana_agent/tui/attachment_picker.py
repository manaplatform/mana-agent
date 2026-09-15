"""File attachment picker modal for Mana Chat TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class FileAttachmentModal(ModalScreen[Path | None]):
    """Modal screen allowing the user to select or enter a file path to attach."""

    CSS = """
    FileAttachmentModal {
        align: center middle;
    }
    #attach-dialog {
        width: 80;
        height: 70%;
        padding: 1 2;
        border: round #6366f1;
        background: #111827;
    }
    #file-path-input {
        margin: 1 0;
    }
    #recent-files-list {
        height: 1fr;
        border: round #334155;
        margin-bottom: 1;
    }
    .modal-actions {
        height: 3;
        align-horizontal: right;
    }
    .modal-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, repo_root: Path | str | None = None) -> None:
        super().__init__()
        self.repo_root = Path(repo_root or ".").resolve()
        self._candidate_files: list[Path] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="attach-dialog"):
            yield Label("📎 Attach file to chat message")
            yield Input(
                placeholder="Enter file path (e.g. ./docs/report.pdf or /path/to/image.png)…",
                id="file-path-input",
            )
            yield Label("Files in workspace:")
            yield ListView(id="recent-files-list")
            yield Static(
                "Supported: Images (PNG, JPG, WEBP, GIF), Documents (PDF, TXT, MD, CSV, JSON), Code, Media.",
                classes="text-muted",
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Attach", id="btn-attach", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        list_view = self.query_one("#recent-files-list", ListView)
        try:
            # Gather up to 30 visible non-hidden files from repo_root
            found: list[Path] = []
            for p in self.repo_root.iterdir():
                if p.name.startswith((".", "_")) or p.name in {"node_modules", "venv", ".venv", "__pycache__"}:
                    continue
                if p.is_file():
                    found.append(p)
                elif p.is_dir():
                    try:
                        for sub in p.iterdir():
                            if not sub.name.startswith(".") and sub.is_file():
                                found.append(sub)
                                if len(found) >= 30:
                                    break
                    except (PermissionError, OSError):
                        pass
                if len(found) >= 30:
                    break
            self._candidate_files = found
            for path in self._candidate_files:
                try:
                    rel = path.relative_to(self.repo_root)
                except ValueError:
                    rel = path
                list_view.append(ListItem(Label(str(rel))))
        except Exception:
            pass
        self.query_one("#file-path-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._confirm_path(event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._candidate_files):
            path = self._candidate_files[idx]
            self.dismiss(path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-attach":
            val = self.query_one("#file-path-input", Input).value
            self._confirm_path(val)
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def _confirm_path(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        p = Path(text).expanduser()
        if not p.is_absolute():
            p = (self.repo_root / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            self.notify(f"File not found: {p}", severity="error")
            return
        self.dismiss(p)
