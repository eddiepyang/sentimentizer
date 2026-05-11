"""Device detection utilities — no module-level torch import.

This module provides ``resolve_device`` which can be imported without
triggering a torch import. Commands that need a concrete device call
``resolve_device`` after they've already imported torch for their own
work, so the import cost is free.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _has_nvidia_libs() -> bool:
    """Check whether NVIDIA CUDA libraries are installed alongside torch.

    When ``pyproject.toml`` pins torch to the CPU-only wheel index (the
    default for CI), ``torch.cuda.is_available()`` returns ``False`` even
    though ``nvidia-*`` packages are present.  Detecting this mismatch
    lets us warn the user that distributed training will run on CPU.
    """
    import importlib.metadata

    for dist in importlib.metadata.distributions():
        if dist.metadata["Name"].lower().startswith("nvidia"):
            return True
    return False


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device string.

    If *device* is anything other than ``"auto"``, it is returned as-is.

    This function imports ``torch`` only when called with ``"auto"``, so
    importing this module never triggers a torch import.

    When torch is the CPU-only build (``+cpu`` suffix) but NVIDIA
    libraries are installed, a warning is logged advising the user to
    install the CUDA variant with ``uv sync --no-sources-package torch``.
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    # CPU-only torch with NVIDIA libraries installed → likely a
    # misconfiguration that will prevent GPU training.
    if "+cpu" in torch.__version__ and _has_nvidia_libs():
        logger.warning(
            "torch %s is CPU-only but NVIDIA CUDA libraries are installed. "
            "Distributed and GPU training will use CPU. "
            "Install the CUDA variant with: uv sync --no-sources-package torch",
            torch.__version__,
        )

    return "cpu"
