"""Existing-SSH enrollment provider marker."""

from __future__ import annotations


class ExistingSSHProvider:
    """Enrollment-only provider; lifecycle methods intentionally do not exist."""

    name = "ssh"
