"""Rate-limit identities for the production API.

The API applies both limits at protected routes: one shared by client IP and
another shared by a fingerprint of the supplied API key.  The raw key never
reaches a limiter key, log entry, or response.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Per-process salt, generated once at import time. Mixing it into the
# fingerprint ensures that even a caller rotating ``X-API-Key`` headers
# cannot enumerate fingerprints in advance (precomputed rainbow tables
# are useless), and that an attacker exfiltrating the salt would have to
# also exfiltrate the rest of the process memory to correlate keys.
_SALT: bytes = os.urandom(32)


def get_api_key_fingerprint(request: Request) -> str:
    """Return a stable, non-reversible limiter identity for the API key."""

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return "anonymous"

    digest = hmac.new(_SALT, api_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"k:{digest}"


def create_limiters() -> tuple[Limiter, Limiter]:
    """Create isolated IP and API-key limiters for an application instance."""

    return (
        Limiter(key_func=get_remote_address),
        Limiter(key_func=get_api_key_fingerprint),
    )
