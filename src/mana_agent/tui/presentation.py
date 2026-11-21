"""Terminal presentation helpers shared by the chat widgets."""

from __future__ import annotations

import os
import sys


def glyph(unicode: str, ascii: str) -> str:
    if os.environ.get("TERM") == "dumb":
        return ascii
    try:
        unicode.encode(getattr(sys.stdout, "encoding", None) or getattr(sys.__stdout__, "encoding", None) or "ascii")
    except UnicodeEncodeError:
        return ascii
    return unicode
