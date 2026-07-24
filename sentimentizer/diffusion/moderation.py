"""Fail-closed client for operator-provided image output moderation."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class ImageModerationError(RuntimeError):
    """Raised when the configured moderation service cannot decide safely."""


@dataclass(frozen=True)
class UnsafeImageError(ImageModerationError):
    """Raised when generated output violates the operator's content policy."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message


class ImageModerationClient:
    """POST generated image bytes to a small, provider-neutral moderation API.

    The endpoint must return a JSON object with ``safe: true`` or
    ``safe: false``. Unsafe responses may additionally include ``code`` and
    ``message``. Transport and schema failures raise ``ImageModerationError``
    so callers can fail closed without releasing an unchecked image.
    """

    def __init__(self, url: str, *, api_key: str = "", timeout_s: float = 10.0) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image_moderation_url must be an http(s) URL")
        if timeout_s <= 0:
            raise ValueError("image_moderation_timeout_s must be positive")
        self.url = url
        self.api_key = api_key
        self.timeout_s = timeout_s

    def check(
        self,
        image: bytes,
        *,
        model: str,
        user: str | None,
        mime_type: str = "image/png",
    ) -> None:
        """Raise unless the moderation endpoint explicitly approves the image."""
        payload = {
            "image_b64": base64.b64encode(image).decode("ascii"),
            "mime_type": mime_type,
            "model": model,
            "user": user,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = _load_response(response.read())
        except urllib.error.HTTPError as exc:
            raise ImageModerationError(
                f"Image moderation returned HTTP {exc.code}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ImageModerationError(
                f"Cannot reach image moderation service: {reason}"
            ) from exc

        if result.get("safe") is True:
            return
        if result.get("safe") is False:
            raise UnsafeImageError(
                code=str(result.get("code") or "content_policy_violation"),
                message=str(result.get("message") or "Generated image was blocked by policy"),
            )
        raise ImageModerationError("Image moderation response must contain a boolean 'safe'")


def _load_response(raw: bytes) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageModerationError("Image moderation returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ImageModerationError("Image moderation returned non-object JSON")
    return result
