"""Lazy public exports for the Sentimentizer Serve package.

Keeping the application import lazy lets ``python -m sentimentizer.serve
--config ...`` select its configuration before Ray deployment decorators read
the module-level config singleton.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SentimentizerDeployment", "app", "main"]


def __getattr__(name: str) -> Any:
    """Load the Serve application only when a public export is requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from sentimentizer.serve.app import SentimentizerDeployment, app, main

    return {
        "SentimentizerDeployment": SentimentizerDeployment,
        "app": app,
        "main": main,
    }[name]
