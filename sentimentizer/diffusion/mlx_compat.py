"""MLX availability detection — no module-level torch import."""

from __future__ import annotations

try:
    import mflux  # noqa: F401

    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False


def is_mlx_device() -> bool:
    """True when running on Apple Silicon with MLX GPU available."""
    try:
        import mlx.core as mx

        return mx.default_device().type == mx.DeviceType.gpu
    except (ImportError, Exception):
        return False
