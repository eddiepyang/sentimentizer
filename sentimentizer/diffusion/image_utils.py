"""Public image utility functions for diffusion pipelines.

These were previously private functions in ``predictor.py`` (prefixed with
``_``).  They are used by both the diffusers predictors and the MLX
predictor, so they belong in a shared public module.
"""

from __future__ import annotations

import base64
import io
import secrets
from typing import Any

import PIL.Image

_REF_MAX_PIXELS = 512 * 512
_MAX_SEED = 2**32 - 1


def decode_b64_image(b64: str, max_pixels: int = _REF_MAX_PIXELS) -> PIL.Image.Image:
    """Decode a base64 image string (raw or data URL) to PIL RGB.

    Raises ValueError on malformed input or images exceeding max_pixels.
    """
    try:
        if b64.startswith("data:image/") and ";base64," in b64:
            b64 = b64.split(";base64,", 1)[1]
        data = base64.b64decode(b64)
        image = PIL.Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"malformed base64 image: {exc}") from exc

    if image.width * image.height > max_pixels:
        raise ValueError(
            f"reference image exceeds max_pixels={max_pixels} ({image.width}x{image.height})"
        )

    return image


def encode_pil(image: Any, format: str = "png") -> bytes:
    """Encode a PIL Image to bytes in the given format."""
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if format == "webp":
        kwargs["quality"] = 85
    elif format == "jpeg":
        kwargs["quality"] = 90
    image.save(buf, format=format.upper(), **kwargs)
    return buf.getvalue()


def b64_encode(data: bytes) -> str:
    """Base64-encode bytes to an ASCII string."""
    return base64.b64encode(data).decode("ascii")


def generate_id() -> str:
    """Generate a unique image ID with ``img_`` prefix."""
    return "img_" + base64.b32encode(secrets.token_bytes(8)).decode("ascii")[:12]
