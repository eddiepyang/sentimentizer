"""Minimal Ray Serve application for BGE-M3 hybrid embeddings."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from sentimentizer import logger
from sentimentizer.embeddings import BGEM3Predictor
from sentimentizer.serve.base import (
    PROMETHEUS_CONTENT_TYPE,
    ServiceMetrics,
    render_service_metrics,
    serve,
)
from sentimentizer.serve.config import cfg
from sentimentizer.serve.embeddings_models import EmbeddingsRequest, EmbeddingsResult

bge_app = FastAPI(
    title="Sentimentizer BGE-M3",
    description="Minimal BGE-M3 hybrid embedding service",
)


@serve.deployment(
    num_replicas=cfg.bge_m3_num_replicas,
    max_ongoing_requests=cfg.bge_m3_max_ongoing_requests,
    ray_actor_options={
        "num_cpus": cfg.bge_m3_num_cpus,
        "num_gpus": cfg.bge_m3_num_gpus,
    },
)
@serve.ingress(bge_app)
class BGEM3OnlyDeployment:
    """Own exactly one BGE-M3 predictor and its HTTP routes."""

    def __init__(self) -> None:
        self._started_at = time.time()
        self._ready = False
        self._metrics = ServiceMetrics(prefix="sentimentizer_bge_m3")
        self._encode_lock = asyncio.Lock()
        self._predictor = BGEM3Predictor(
            model_id=cfg.bge_m3_model_id,
            device=cfg.embeddings_device,
            use_fp16=cfg.bge_m3_use_fp16,
            batch_size=cfg.bge_m3_batch_size,
        )
        self._ready = True

    @bge_app.post("/v1/embeddings", response_model=EmbeddingsResult)
    async def embeddings(self, body: EmbeddingsRequest) -> dict[str, Any]:
        """Return BGE-M3 dense and learned-sparse vectors in input order."""
        started_at = time.perf_counter()
        try:
            async with self._encode_lock:
                vectors = await asyncio.to_thread(self._predictor.encode, body.texts)
        except Exception:
            self._metrics.record_request(time.perf_counter() - started_at, error=True)
            raise
        self._metrics.record_request(time.perf_counter() - started_at)
        return {
            "backend": "flag_embedding",
            "model": cfg.bge_m3_model_id,
            "vectors": vectors,
        }

    @bge_app.get("/health/live")
    async def health_live(self) -> dict[str, str | float]:
        """Return liveness without invoking the model."""
        return {
            "status": "alive",
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    @bge_app.get("/health/ready", response_model=None)
    async def health_ready(self) -> dict[str, str] | JSONResponse:
        """Return readiness after the BGE-M3 constructor completes."""
        body = {"status": "ready" if self._ready else "not_ready", "model": cfg.bge_m3_model_id}
        if not self._ready:
            return JSONResponse(status_code=503, content=body)
        return body

    @bge_app.get("/health", response_model=None)
    async def health(self) -> dict[str, str] | JSONResponse:
        """Return the compatibility readiness response."""
        return await self.health_ready()

    @bge_app.get("/metrics", response_class=Response)
    async def metrics(self) -> Response:
        """Return per-replica Prometheus service and inference metrics."""
        body = render_service_metrics(
            (self._metrics,),
            service_prefix="sentimentizer_bge_m3_service",
            ready=self._ready,
            uptime_s=time.time() - self._started_at,
        )
        return Response(
            content=body,
            headers={"Content-Type": PROMETHEUS_CONTENT_TYPE},
        )


def initialize_ray(
    ray_module: Any,
    object_store_memory_mb: int,
    python_executable: str = sys.executable,
) -> None:
    """Initialize Ray with the production object-store allocation."""
    if object_store_memory_mb < 1:
        raise ValueError("object_store_memory_mb must be greater than zero")
    ray_module.init(
        ignore_reinit_error=True,
        namespace="sentimentizer-bge",
        object_store_memory=object_store_memory_mb * 1024 * 1024,
        runtime_env={"py_executable": python_executable},
    )


def main(
    host: str | None = None,
    port: int | None = None,
    object_store_memory_mb: int | None = None,
) -> None:
    """Start the BGE-M3-only Ray Serve application."""
    host = cfg.serve_host if host is None else host
    port = cfg.serve_port if port is None else port
    object_store_memory_mb = (
        cfg.ray_object_store_memory_mb
        if object_store_memory_mb is None
        else object_store_memory_mb
    )
    os.environ.setdefault("RAY_ENABLE_RUNTIME_ENV_HOOK", "1")
    os.environ.setdefault("RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES", "")

    import ray
    from ray.serve.config import HTTPOptions

    initialize_ray(ray, object_store_memory_mb)
    serve.start(
        http_options=HTTPOptions(
            host=host,
            port=port,
            request_timeout_s=cfg.request_timeout_s,
        )
    )
    serve.run(
        BGEM3OnlyDeployment.bind(),
        name="bge-m3",
        route_prefix="/",
    )

    shutdown_event = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal {signum}"
        logger.info("received_shutdown_signal", signal=signame)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(
        "Sentimentizer BGE-M3 Serve running",
        host=host,
        port=port,
        object_store_memory_mb=object_store_memory_mb,
    )
    shutdown_event.wait()
    logger.info("Shutting down BGE-M3 Serve")
    serve.shutdown()
    ray.shutdown()
