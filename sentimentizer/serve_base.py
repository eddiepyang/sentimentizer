"""Shared infrastructure for Ray Serve deployments.

Provides common Metrics, health state, and response schema utilities
used by both the sentiment analysis and router serve endpoints.
"""

import threading
from typing import Any

from starlette.responses import JSONResponse

try:
    from ray import serve

    RAY_SERVE_AVAILABLE = True
except ImportError:
    RAY_SERVE_AVAILABLE = False

    class _DummyServe:
        def deployment(self, *args: Any, **kwargs: Any) -> Any:
            return lambda cls: cls

    serve = _DummyServe()


from sentimentizer import logger


class ServiceMetrics:
    """Thread-safe request metrics collector with Prometheus-compatible output.

    Used by both SentimentDeployment and RouterDeployment to track
    request counts, errors, and latency.
    """

    def __init__(self, prefix: str = "sentimentizer") -> None:
        self._prefix = prefix
        self._lock = threading.Lock()
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_s: float = 0.0

    def record_request(self, latency_s: float, error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_s += latency_s
            if error:
                self.error_count += 1

    def to_prometheus(self) -> str:
        with self._lock:
            lines = [
                f"# HELP {self._prefix}_request_total Total requests",
                f"# TYPE {self._prefix}_request_total counter",
                f"{self._prefix}_request_total {self.request_count}",
                f"# HELP {self._prefix}_error_total Total errors",
                f"# TYPE {self._prefix}_error_total counter",
                f"{self._prefix}_error_total {self.error_count}",
                f"# HELP {self._prefix}_latency_seconds_total Cumulative latency in seconds",
                f"# TYPE {self._prefix}_latency_seconds_total counter",
                f"{self._prefix}_latency_seconds_total {self.total_latency_s:.6f}",
            ]
            return "\n".join(lines) + "\n"

    @property
    def avg_latency_s(self) -> float:
        with self._lock:
            return self.total_latency_s / self.request_count if self.request_count else 0


# Singleton instances — one for sentiment, one for router
sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
router_metrics = ServiceMetrics(prefix="router")


def build_predict_response(
    text: str,
    prediction: dict[str, Any],
    latency_s: float,
    metrics: ServiceMetrics,
    log_name: str = "prediction",
    **log_extra: Any,
) -> JSONResponse:
    """Build a unified prediction response.

    Returns JSONResponse with schema:
    {"text": str, "prediction": {...}, "latency_s": float}
    """
    metrics.record_request(latency_s)
    logger.info(
        f"{log_name} completed",
        input_length=len(text),
        latency_s=f"{latency_s:.4f}",
        **log_extra,
    )
    return JSONResponse(
        {
            "text": text,
            "prediction": prediction,
            "latency_s": round(latency_s, 4),
        }
    )


def build_batch_response(
    results: list[dict],
    latency_s: float,
    metrics: ServiceMetrics,
    log_name: str = "batch prediction",
    **log_extra: Any,
) -> JSONResponse:
    """Build a unified batch response.

    Returns JSONResponse with schema:
    {"results": [...], "count": int, "latency_s": float}
    """
    metrics.record_request(latency_s)
    logger.info(
        f"{log_name} completed",
        batch_size=len(results),
        latency_s=f"{latency_s:.4f}",
        **log_extra,
    )
    return JSONResponse(
        {
            "results": results,
            "count": len(results),
            "latency_s": round(latency_s, 4),
        }
    )


def build_error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Build an error response with the standard schema."""
    return JSONResponse({"error": message}, status_code=status_code)


def build_health_response(loaded: bool, **extra: Any) -> JSONResponse:
    """Build a health check response."""
    status = "healthy" if loaded else "unhealthy"
    code = 200 if loaded else 503
    return JSONResponse({"status": status, **extra}, status_code=code)


def build_metrics_response(metrics: ServiceMetrics) -> JSONResponse:
    """Build a metrics response."""
    return JSONResponse(
        {
            "prometheus": metrics.to_prometheus(),
            "request_count": metrics.request_count,
            "error_count": metrics.error_count,
            "avg_latency_s": metrics.avg_latency_s,
        }
    )
