"""Compatibility shims for third-party library issues.

These must be imported BEFORE the problematic library to ensure
the monkey-patches are applied at import time.
"""


def patch_transformers_default_logdir() -> None:
    """Inject ``default_logdir`` into ``transformers.training_args`` if missing.

    ``setfit 1.1.x`` imports ``default_logdir`` from
    ``transformers.training_args``, but ``transformers 5.x`` removed this
    function. This shim injects a compatible stub so that ``setfit`` can
    be imported successfully with ``transformers>=5.0``.

    Must be called BEFORE ``import setfit`` — otherwise the ImportError
    is raised at module load time.
    """
    import importlib

    _module = importlib.import_module("transformers.training_args")
    if not hasattr(_module, "default_logdir"):
        import os

        def _default_logdir() -> str:
            """Return the default logging directory (compatible with transformers<5)."""
            return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")

        _module.default_logdir = _default_logdir


# Apply the shim at module import time so that any subsequent
# ``import setfit`` will succeed.
patch_transformers_default_logdir()
