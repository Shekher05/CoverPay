"""API access control: an optional shared-key gate and a small rate limiter.

Both are deliberately minimal. The key check is a constant-time compare against
`settings.api_key`; when that is unset the dependency is a no-op, so local
development and the test suite run unchanged. The rate limiter is an in-process
fixed window - see the ceiling noted on `config.Settings.rate_limit_per_minute`.

What this is *not*: an identity system. A valid key does not identify a
merchant, so an endpoint that accepts a `merchant_id` still cannot trust it to
scope data to the caller. That needs a real principal model and is out of scope.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict

from fastapi import Header, HTTPException, Request

from config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency. No-op when no key is configured; otherwise the
    request must carry a matching `X-API-Key` header."""
    expected = settings.api_key
    if not expected:
        return
    # Compare as bytes: `compare_digest` raises TypeError on a non-ASCII str,
    # which would surface as a 500 instead of a clean 401.
    supplied = (x_api_key or "").encode("utf-8", "ignore")
    if not x_api_key or not secrets.compare_digest(supplied, expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


class _FixedWindow:
    """Per-caller request count over a rolling 60-second window.

    ponytail: the caller dict is never swept, so a process seeing millions of
    distinct unauthenticated client IPs would leak memory. Fine behind the API
    key; add an LRU or periodic sweep before exposing it unauthenticated.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, caller: str) -> None:
        now = time.monotonic()
        # Held across the read-modify-write: FastAPI runs sync dependencies in a
        # threadpool, so an unlocked counter loses increments under load.
        with self._lock:
            recent = [t for t in self._hits[caller] if now - t < 60.0]
            if len(recent) >= self._limit:
                raise HTTPException(
                    status_code=429, detail="Rate limit exceeded, retry in a minute"
                )
            recent.append(now)
            self._hits[caller] = recent

    def reset(self) -> None:
        """Clear all windows. For tests, which share one process-global limiter
        across a wall-clock window."""
        with self._lock:
            self._hits.clear()


_limiter = _FixedWindow(settings.rate_limit_per_minute)


def _client_ip(request: Request) -> str:
    """The caller's real IP.

    Behind a reverse proxy (every PaaS: Render, Railway, Fly, nginx) the socket
    peer is the proxy, so `request.client.host` is one shared value for every
    visitor - which would put the whole site in a single rate-limit bucket. The
    proxy forwards the original client as the left-most entry of
    `X-Forwarded-For`, so prefer that.

    ponytail: X-Forwarded-For is client-set and only trustworthy because a proxy
    we control overwrites/prepends it. A caller hitting the app directly (no
    proxy) could spoof it to dodge their own limit - acceptable here, since the
    limiter is a courtesy throttle and real spend protection is API_KEY plus the
    provider-side quota. If the app is ever exposed with no proxy in front, key
    on request.client.host only.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency for the expensive endpoints (assistant, CSV upload).
    Counts against the API key when present, else the caller's real IP."""
    caller = x_api_key or _client_ip(request)
    _limiter.check(caller)
