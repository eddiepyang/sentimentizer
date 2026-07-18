"""Shared infrastructure for Ray Serve deployments.

Provides ``ServiceMetrics`` for request/latency tracking and a
``serve`` import helper that degrades gracefully when Ray is not installed.
"""

import threading
from collections.abc import Iterable
from typing import Any

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class _DummyHandle:
    """Stub for Ray Serve deployment handles (returned by Deployment.bind())."""

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Ray Serve is not available. Install with: pip install 'ray[serve]'")

    def options(self, **kwargs: Any) -> "_DummyHandle":
        return self


class _DummyServe:
    def deployment(self, *args: Any, **kwargs: Any) -> Any:
        def decorator(cls: Any) -> Any:
            if not hasattr(cls, "bind"):
                cls.bind = classmethod(lambda cls_, *a, **kw: _DummyHandle())
            return cls

        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator

    def ingress(self, *args: Any, **kwargs: Any) -> Any:
        def decorator(cls: Any) -> Any:
            return cls

        if len(args) == 1 and callable(args[0]) and not hasattr(args[0], "routes"):
            return decorator(args[0])
        return decorator

    def start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, *args: Any, **kwargs: Any) -> None:
        pass

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        pass


try:
    from ray import serve

    RAY_SERVE_AVAILABLE = True
except ImportError:
    RAY_SERVE_AVAILABLE = False
    serve = _DummyServe()


class ServiceMetrics:
    """Thread-safe request metrics collector with Prometheus-compatible output.

    Picklable for Ray Serve serialization — the lock is recreated on
    deserialization since it has no meaningful state to preserve.
    """

    def __init__(self, prefix: str = "sentimentizer") -> None:
        self._prefix = prefix
        self._lock = threading.Lock()
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_s: float = 0.0

    def __getstate__(self) -> dict:
        return {
            "_prefix": self._prefix,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_latency_s": self.total_latency_s,
        }

    def __setstate__(self, state: dict) -> None:
        self._prefix = state["_prefix"]
        self._lock = threading.Lock()
        self.request_count = state["request_count"]
        self.error_count = state["error_count"]
        self.total_latency_s = state["total_latency_s"]

    def record_request(self, latency_s: float, error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_s += latency_s
            if error:
                self.error_count += 1

    def to_prometheus(self) -> str:
        """Generate Prometheus exposition format text."""
        p = self._prefix
        with self._lock:
            metrics = [
                (f"{p}_request_total", self.request_count, "Total requests"),
                (f"{p}_error_total", self.error_count, "Total errors"),
                (
                    f"{p}_latency_seconds_total",
                    f"{self.total_latency_s:.6f}",
                    "Cumulative latency in seconds",
                ),
            ]
            lines = []
            for name, value, help_text in metrics:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            return "\n".join(lines) + "\n"

    @property
    def avg_latency_s(self) -> float:
        with self._lock:
            return self.total_latency_s / self.request_count if self.request_count else 0.0


def render_service_metrics(
    collectors: Iterable[ServiceMetrics],
    *,
    service_prefix: str,
    ready: bool,
    uptime_s: float,
) -> str:
    """Render service state and request collectors in Prometheus text format."""
    lines = [
        f"# HELP {service_prefix}_ready Whether the service is ready to accept traffic",
        f"# TYPE {service_prefix}_ready gauge",
        f"{service_prefix}_ready {int(ready)}",
        f"# HELP {service_prefix}_uptime_seconds Service uptime in seconds",
        f"# TYPE {service_prefix}_uptime_seconds gauge",
        f"{service_prefix}_uptime_seconds {uptime_s:.6f}",
    ]
    lines.extend(collector.to_prometheus().rstrip() for collector in collectors)
    return "\n".join(lines) + "\n"
