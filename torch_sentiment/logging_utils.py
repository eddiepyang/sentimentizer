import io
import logging
import sys
import structlog
import time
import psutil

from functools import wraps
from typing import Callable, TextIO


def new_logger(level: int = 20, output: TextIO = sys.stderr) -> structlog.PrintLogger:
    """
    creates instance of struct logger
    """
    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.threadlocal.merge_threadlocal_context,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=output),
    )
    return structlog.getLogger(__name__)


logger = new_logger(logging.INFO)


def time_decorator_factory(optional_text: str = ""):
    def time_decorator(func) -> Callable:
        """logs time stats of function"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            ts = time.perf_counter()
            result = func(*args, **kwargs)
            te = time.perf_counter()
            event = "function completed successfully"
            if optional_text:
                event = f"function completed successfully: {optional_text}"
            logger.info(
                event,
                function=func.__name__,
                run_time=f"{te-ts: 2.4f} seconds",
                available_memory=f"{psutil.virtual_memory().available/1024**3: .2f} GBs",
                free_memory=f"{psutil.virtual_memory().free/1024**3: .2f} GBs",
                used_memory=f"{psutil.virtual_memory().used/1024**3: .2f} GBs",
            )
            return result

        return wrapper

    return time_decorator


def time_decorator(func) -> Callable:
    """logs time stats of function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        event = "function completed successfully"
        logger.info(
            event,
            function=func.__name__,
            run_time=f"{te-ts: 2.4f} seconds",
            available_memory=f"{psutil.virtual_memory().available/1024**3: .2f} GBs",
            free_memory=f"{psutil.virtual_memory().free/1024**3: .2f} GBs",
            used_memory=f"{psutil.virtual_memory().used/1024**3: .2f} GBs",
        )
        return result

    return wrapper
