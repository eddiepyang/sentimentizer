"""Pipeline lifecycle: environment setup, Ray init, cleanup, and shared state.

This module has module-level side effects that execute at import time:
  - load_dotenv() is called
  - SENTIMENTIZER_* / RAY_* env vars are set as defaults
  - atexit handlers are registered for CUDA and Ray cleanup
  - SIGINT handler is registered for graceful shutdown

IMPORTANT: This module MUST be imported before any ray/torch usage.
The env var defaults and cleanup handlers are required for correct behavior.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from sentimentizer import new_logger

_INFO = logging.INFO
logger = new_logger(_INFO)

# ── Environment setup ───────────────────────────────────────────

load_dotenv()

for _key, _val in {
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
    "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.75",
    "RAY_GRAFANA_HOST": "http://localhost:3000",
    "RAY_PROMETHEUS_HOST": "http://localhost:9090",
    # Reduce CUDA memory fragmentation — PyTorch's default buddy allocator
    # can leave many small holes; expandable_segments lets the allocator
    # merge adjacent free regions into larger ones, reducing OOM from
    # fragmentation (the "Tried to allocate X MiB but only Y MiB free" case
    # where total free memory is sufficient but no single region is large
    # enough).  Safe on all platforms; no-op when CUDA is not available.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(_key, _val)

# Note: do NOT call logging.basicConfig() here or in the cli callback.
# Logging is configured by structlog in sentimentizer/__init__.py at import
# time; basicConfig wouldn't reach structlog's logger factory and would
# silently shadow the existing config for any stdlib-logging consumers.

# ── Cleanup handlers ────────────────────────────────────────────

# Ray stores session data in /tmp/ray/session_<timestamp>_<id>/.
# Each session can consume 5+ GB. When Ray is not shut down cleanly
# (e.g., Ctrl-C, crash, or test runner killing the process), these
# directories accumulate and fill the disk.
_RAY_SESSION_DIR = Path("/tmp/ray")


def _cuda_cleanup() -> None:
    """Release cached CUDA memory. Safe even when torch was never imported."""
    if "torch" not in sys.modules:
        return
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            # CUDA context may already be torn down during shutdown
            pass


def _cleanup_stale_ray_sessions() -> None:
    """Remove stale Ray session directories from /tmp/ray.

    Called before ray.init() to free disk space from previous runs.
    Only removes sessions older than 1 hour to avoid deleting active sessions.
    """
    if not _RAY_SESSION_DIR.exists():
        return

    import time

    current_time = time.time()
    removed = 0
    freed_gb = 0.0

    for session_dir in _RAY_SESSION_DIR.iterdir():
        if session_dir.name == "session_latest":
            continue
        if session_dir.is_dir() and session_dir.name.startswith("session_"):
            try:
                age_hours = (current_time - os.path.getmtime(session_dir)) / 3600
                if age_hours > 1:  # older than 1 hour
                    size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
                    shutil.rmtree(session_dir, ignore_errors=True)
                    freed_gb += size / (1024**3)
                    removed += 1
            except OSError:
                pass

    if removed > 0:
        with contextlib.suppress(Exception):
            logger.info(  # type: ignore[call-arg]
                "cleaned_stale_ray_sessions",
                removed=removed,
                freed_gb=round(freed_gb, 1),
            )


def _ray_cleanup() -> None:
    """Shut down Ray and clean up stale session temp files.

    Safe even when ray was never imported.
    """
    if "ray" not in sys.modules:
        return
    import ray

    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    # Reuse the dedicated cleanup function instead of reimplementing the walk
    _cleanup_stale_ray_sessions()


def _sigint_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl-C by cleaning up CUDA and Ray before re-raising."""
    logger.info("Received SIGINT, cleaning up...")
    _cuda_cleanup()
    _ray_cleanup()
    raise KeyboardInterrupt


atexit.register(_cuda_cleanup)
atexit.register(_ray_cleanup)
signal.signal(signal.SIGINT, _sigint_handler)

# ── Ray bootstrap ───────────────────────────────────────────────


def is_ray_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("ray") is not None
    except Exception:
        return False


def _kill_stale_ray_processes() -> None:
    """Kill orphaned raylet/GCS processes left by a previous Ctrl-C.

    After a hard kill (SIGKILL or uncaught SIGINT) Ray processes can remain
    alive, holding port 6379 or the GCS lock and causing the next ray.init()
    to time out with "raylet failed to startup".  We send SIGTERM first,
    wait briefly, then SIGKILL stragglers.
    """
    import subprocess
    import time

    TARGETS = ["raylet", "gcs_server", "ray::IDLE"]
    for name in TARGETS:
        with contextlib.suppress(FileNotFoundError):
            subprocess.run(["pkill", "-TERM", "-f", name], capture_output=True)
    time.sleep(1)
    for name in TARGETS:
        with contextlib.suppress(FileNotFoundError):
            subprocess.run(["pkill", "-KILL", "-f", name], capture_output=True)


def _ensure_ray_initialized() -> bool:
    """Initialize Ray with metrics port and runtime_env if not already running.

    Every ``run_*`` helper that touches ``ray.data`` or ``ray.train`` calls
    this as its first line.  Click command shells stay thin (one-line
    forwarders) and don't call it themselves — that way you can't
    accidentally use a helper from another helper or a test without
    bringing Ray with it.

    Returns True if Ray was successfully initialized (or was already running),
    and False if the `ray` package is not installed.

    On startup timeout (stale GCS lock from a previous Ctrl-C), kills stale
    Ray processes, clears /tmp/ray/, and retries once automatically.
    """
    try:
        import ray
    except ImportError:
        logger.info("Ray is not installed, skipping Ray initialization.")
        return False

    if ray.is_initialized():
        return True
    # Suppress smart_open verbose logging that floods Ray worker output.
    import logging

    from sentimentizer.env import ensure_nvidia_ld_library_path

    logging.getLogger("smart_open").setLevel(logging.WARNING)

    ld_path = ensure_nvidia_ld_library_path()
    runtime_env: dict[str, Any] = {}
    if ld_path:
        runtime_env["env_vars"] = {"LD_LIBRARY_PATH": ld_path}
    _cleanup_stale_ray_sessions()

    try:
        ray.init(_metrics_export_port=8080, runtime_env=runtime_env)
    except Exception as exc:
        # "timed out during startup" / "raylet failed to startup" are symptoms
        # of stale processes from a previous hard kill.  Kill them and retry.
        msg = str(exc).lower()
        if "timed out" in msg or "raylet" in msg or "gcs" in msg or "startup" in msg:
            logger.warning(  # type: ignore[call-arg]
                "ray_init_timeout_retrying",
                message=(
                    "ray.init() timed out — killing stale Ray processes and retrying. "
                    f"Original error: {exc}"
                ),
            )
            _kill_stale_ray_processes()
            _cleanup_stale_ray_sessions()
            ray.init(_metrics_export_port=8080, runtime_env=runtime_env)
        else:
            raise
    return True


# ── Shared state ────────────────────────────────────────────────


@dataclass
class State:
    """Carries CLI flags through the pipeline."""

    model: str
    device: str  # raw value: "auto" | "cuda" | "mps" | "cpu" — resolve lazily
    run_type: str  # "new" | "update"
