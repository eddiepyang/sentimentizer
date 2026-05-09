"""Device detection utilities — no module-level torch import.

This module provides ``resolve_device`` which can be imported without
triggering a torch import. Commands that need a concrete device call
``resolve_device`` after they've already imported torch for their own
work, so the import cost is free.
"""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device string.

    If *device* is anything other than ``"auto"``, it is returned as-is.

    This function imports ``torch`` only when called with ``"auto"``, so
    importing this module never triggers a torch import.
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
