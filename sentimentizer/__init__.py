import logging
import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TextIO

import psutil
import structlog

# Disable Ray's automatic uv runtime environment to prevent VIRTUAL_ENV warnings
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

file_path = Path(__file__)
root = file_path.parent.parent.absolute()


def new_logger(level: int = 20, output: TextIO = sys.stderr) -> Any:
    """Creates a configured structlog logger.

    Returns Any because structlog's bound logger accepts arbitrary
    keyword arguments for event key-value pairs, which static type
    checkers cannot express.
    """
    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=output),
    )
    return structlog.getLogger(__name__)


logger: Any = new_logger(logging.INFO)


def time_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """logs time stats of function"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        event = "function completed successfully"
        logger.info(
            event,
            function=func.__name__,
            run_time=f"{te - ts: 2.4f} seconds",
            available_memory=f"{psutil.virtual_memory().available / 1024**3: .2f} GBs",
            free_memory=f"{psutil.virtual_memory().free / 1024**3: .2f} GBs",
            used_memory=f"{psutil.virtual_memory().used / 1024**3: .2f} GBs",
        )
        return result

    return wrapper
