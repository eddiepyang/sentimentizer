import os
import sys


def get_nvidia_ld_library_path() -> str:
    """Detect NVIDIA CUDA library dirs in the venv and return them as a path string."""
    paths = []
    for _sp in sys.path:
        _nvidia_pkg = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_pkg):
            for _nvidia_dir in os.listdir(_nvidia_pkg):
                _lib_dir = os.path.join(_nvidia_pkg, _nvidia_dir, "lib")
                if os.path.isdir(_lib_dir):
                    paths.append(_lib_dir)
            break
    return ":".join(paths)


def ensure_nvidia_ld_library_path() -> str:
    """Ensure NVIDIA CUDA library dirs are added to LD_LIBRARY_PATH.

    Required for Ray workers that need torch's CUDA dependencies but don't
    inherit the driver process's environment. Returns the updated LD_LIBRARY_PATH.
    """
    nvidia_paths = get_nvidia_ld_library_path()
    if not nvidia_paths:
        return os.environ.get("LD_LIBRARY_PATH", "")

    existing = os.environ.get("LD_LIBRARY_PATH", "")

    # Only add paths that aren't already there
    for path in nvidia_paths.split(":"):
        if path not in existing:
            existing = path + (":" + existing if existing else "")

    os.environ["LD_LIBRARY_PATH"] = existing
    return existing
