from __future__ import annotations

import hashlib
import hmac


def verify_signature(raw_body: bytes, signature_header: str | None, secret: str) -> bool:
    """Validate GitHub's sha256 signature without decoding or changing the body."""
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    supplied = signature_header.removeprefix("sha256=")
    if len(supplied) != 64:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, supplied.lower())
