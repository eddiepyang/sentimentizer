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

# Enable Ray Data rich progress bars and suppress the "new progress UI" info message.
# These must be set before ray.data.context is imported so the defaults are picked up.
# See https://docs.ray.io/en/2.55.1/data/api/doc/ray.data.DataContext.html
os.environ["RAY_DATA_ENABLE_RICH_PROGRESS_BARS"] = "1"
os.environ["RAY_TQDM"] = "0"

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


def configured_log_level(value: str | None = None) -> int:
    """Resolve SENTIMENTIZER_LOG_LEVEL, falling back safely to INFO."""
    name = (value if value is not None else os.getenv("SENTIMENTIZER_LOG_LEVEL", "INFO")).upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


logger: Any = new_logger(configured_log_level())


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
