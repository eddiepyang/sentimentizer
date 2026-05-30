"""Middleware for image generation routes: auth, rate limiting, idempotency, safety.

All middleware is scoped to image generation routes via FastAPI
``dependencies=[...]``. Existing sentiment endpoints remain
unauthenticated and unaffected.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, Response

from sentimentizer import logger
from sentimentizer.safety import is_safe

_valid_keys_cache: frozenset[str] | None = None


def _load_valid_keys() -> frozenset[str]:
    """Load valid API keys from the ServeConfig (written to env at startup).

    Cached on first call; refresh by setting the module-global
    ``_valid_keys_cache`` back to ``None``.
    """
    global _valid_keys_cache
    if _valid_keys_cache is None:
        valid_keys_str = os.environ.get("SENTIMENTIZER_API_KEYS", "")
        _valid_keys_cache = frozenset(k.strip() for k in valid_keys_str.split(",") if k.strip())
    return _valid_keys_cache


def require_api_key(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validate Bearer token against SENTIMENTIZER_API_KEYS.

    Returns the validated API key string for downstream use.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_api_key", "message": "Missing Authorization header"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid Authorization header format",
            },
        )

    token = parts[1].strip()
    valid_keys = _load_valid_keys()

    if token not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_api_key", "message": "Invalid API key"},
        )

    logger.debug("api_key_authenticated", key_prefix=token[:8])
    return token


@dataclass
class RateLimitState:
    limit: int
    remaining: int
    reset_at: int


class RateLimiter:
    """In-memory token bucket rate limiter per API key.

    Per-replica only — scaling out weakens the guarantee.
    Acceptable for v1; cluster-wide enforcement (Redis) is P2.
    """

    def __init__(self, requests_per_min: int = 60, burst: int = 10) -> None:
        self._rpm = requests_per_min
        self._burst = burst
        self._buckets: dict[str, _TokenBucket] = {}

    def check(self, key: str) -> RateLimitState:
        now = time.time()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _TokenBucket(self._rpm, self._burst)
            self._buckets[key] = bucket

        # Evict stale buckets periodically (amortized O(1))
        if len(self._buckets) > 100:
            self._evict_stale(now)

        allowed = bucket.consume(now)
        reset_at = int(bucket.reset_at(now))

        if allowed:
            return RateLimitState(
                limit=self._rpm,
                remaining=bucket.remaining(now),
                reset_at=reset_at,
            )

        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limit_exceeded", "message": "Rate limit exceeded"},
            headers={"Retry-After": str(bucket.retry_after(now))},
        )

    def state(self, key: str) -> RateLimitState:
        now = time.time()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _TokenBucket(self._rpm, self._burst)
            self._buckets[key] = bucket
        return RateLimitState(
            limit=self._rpm,
            remaining=bucket.remaining(now),
            reset_at=int(bucket.reset_at(now)),
        )

    def _evict_stale(self, now: float) -> None:
        """Drop buckets whose last refill was more than 2x their full-refill window."""
        max_age = max(120, 2 * self._burst * 60 // max(self._rpm, 1))
        stale_keys = [
            k
            for k, b in self._buckets.items()
            if b._last_refill and (now - b._last_refill) > max_age
        ]
        for k in stale_keys:
            del self._buckets[k]


class _TokenBucket:
    def __init__(self, rpm: int, burst: int) -> None:
        self._rpm = rpm
        self._burst = burst
        self._tokens: float = float(burst)
        self._last_refill: float = 0.0  # set on first access

    def _refill(self, now: float) -> None:
        if self._last_refill == 0.0:
            self._last_refill = now
            return
        elapsed = now - self._last_refill
        added = elapsed * (self._rpm / 60.0)
        self._tokens = min(self._burst, self._tokens + added)
        self._last_refill = now

    def consume(self, now: float) -> bool:
        self._refill(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def remaining(self, now: float) -> int:
        self._refill(now)
        return int(self._tokens)

    def reset_at(self, now: float) -> float:
        self._refill(now)
        if self._tokens >= 1.0:
            return now + 60
        wait = (1.0 - self._tokens) / (self._rpm / 60.0)
        return now + wait

    def retry_after(self, now: float) -> int:
        self._refill(now)
        if self._tokens >= 1.0:
            return 1
        wait = (1.0 - self._tokens) / (self._rpm / 60.0)
        return max(1, int(wait) + 1)


_limiter: RateLimiter | None = None


def _get_limiter() -> RateLimiter:
    """Get or initialize the global RateLimiter lazily from ServeConfig."""
    global _limiter
    if _limiter is None:
        from sentimentizer.serve.config import load_serve_config

        cfg = load_serve_config()
        _limiter = RateLimiter(
            requests_per_min=cfg.rate_limit_per_min,
            burst=cfg.rate_limit_burst,
        )
    return _limiter


def rate_limit(
    request: Request,
    response: Response,
    api_key: str = Depends(require_api_key),
) -> None:
    """FastAPI dependency: enforce per-key rate limit and inject headers."""
    limiter = _get_limiter()
    state = limiter.check(api_key)
    response.headers["X-RateLimit-Limit"] = str(state.limit)
    response.headers["X-RateLimit-Remaining"] = str(state.remaining)
    response.headers["X-RateLimit-Reset"] = str(state.reset_at)


class IdempotencyCache:
    """In-memory idempotency key cache with TTL.

    Per-replica only — a request retried against a different replica
    won't hit. Acceptable for v1; swap to Redis if needed.
    """

    def __init__(self, ttl_s: int = 600) -> None:
        self._ttl_s = ttl_s
        self._cache: dict[tuple[str, str], tuple[Any, str, float]] = {}

    def get(self, api_key: str, key: str) -> Any | None:
        cache_key = (api_key, key)
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        body, _body_hash, expires_at = entry
        if time.time() > expires_at:
            del self._cache[cache_key]
            return None
        return body

    def put(self, api_key: str, key: str, body: Any, request_body_hash: str = "") -> None:
        cache_key = (api_key, key)
        self._cache[cache_key] = (body, request_body_hash, time.time() + self._ttl_s)
        # Evict expired entries periodically (amortized O(1))
        if len(self._cache) > 100:
            self.reap()

    def check_conflict(self, api_key: str, key: str, request_body_hash: str) -> None:
        cache_key = (api_key, key)
        entry = self._cache.get(cache_key)
        if entry is None:
            return
        _body, stored_hash, _expires_at = entry
        if stored_hash and stored_hash != request_body_hash:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_key_conflict",
                    "message": "Idempotency key already used with a different request body",
                },
            )

    def reap(self) -> int:
        now = time.time()
        expired = [k for k, (_, _, exp) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]
        return len(expired)


_IDEM_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def idempotent(
    request: Request,
    api_key: str = Depends(require_api_key),
) -> str | None:
    """FastAPI dependency: read optional Idempotency-Key header.

    Returns the key string if present and valid, None otherwise.
    The handler is responsible for calling ``IdempotencyCache.put()``
    after a successful response.

    Conflict detection (same key + different body → 409) is handled
    inside the route body rather than this dependency because the
    request body is not available in dependency context.
    """
    raw = request.headers.get("Idempotency-Key")
    if raw is None:
        return None
    if not _IDEM_KEY_RE.match(raw):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "bad_request",
                "message": "Idempotency-Key must be 1-128 chars: alnum, dash, underscore",
            },
        )
    return raw


def check_prompt_safety(prompt: str) -> None:
    """Validate prompt against safety rules.

    Raises HTTPException(400) with structured error code on violation.
    Used inside the route body (not as a dependency) because it
    inspects the request payload.
    """
    safe, code, message = is_safe(prompt)
    if not safe:
        raise HTTPException(status_code=400, detail={"code": code, "message": message})
