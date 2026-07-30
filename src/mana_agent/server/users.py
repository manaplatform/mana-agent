"""Linux user and authorized-key administration builders."""

from __future__ import annotations

import re


_USER = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def user_create_argv(username: str, *, shell: str = "/bin/bash") -> list[str]:
    if not _USER.fullmatch(username) or not shell.startswith("/"):
        raise ValueError("A valid exact username and absolute shell path are required.")
    return ["sudo", "useradd", "--create-home", "--shell", shell, "--", username]


def user_delete_argv(username: str, *, remove_home: bool = False) -> list[str]:
    if not _USER.fullmatch(username):
        raise ValueError("A valid exact username is required.")
    return ["sudo", "userdel", *( ["--remove"] if remove_home else []), "--", username]


def validate_public_key(public_key: str) -> str:
    value = public_key.strip()
    if not value.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")) or "\n" in value:
        raise ValueError("Only one valid OpenSSH public-key line may be installed.")
    return value
