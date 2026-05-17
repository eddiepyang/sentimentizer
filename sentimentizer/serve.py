"""Serve the Sentimentizer pipeline via Ray Serve.

Provides a unified REST API for:
  - Sentiment analysis (encoder model by default)
  - Review routing (Dietary, Service, General categories)

Uses ``serve.batch`` to auto-batch individual ``/predict`` and
``/router/predict`` requests into efficient forward passes, and
``asyncio.to_thread()`` to keep sync model inference off the event loop.

Uses FastAPI for HTTP routing via ``@serve.ingress(app)`` with
decorator-based route registration.

Configuration is loaded from ``serve_config.yaml`` with environment
variable overrides. See ``ServeConfig`` for all settings and their
corresponding env vars.

Usage:
    ray serve run sentimentizer.serve:app
    # or programmatically:
    python -m sentimentizer.serve

Endpoints:
  Sentiment analysis:
    POST /predict         -- Classify a single text (auto-batched)
    POST /batch           -- Classify multiple texts (single forward pass)
    POST /tokenize        -- Tokenize text without inference
    GET  /models          -- Sentiment model metadata
    GET  /health          -- Health check
    GET  /metrics          -- Request metrics

  Router (review categorization):
    POST /router/predict  -- Route a single text (auto-batched)
    POST /router/batch    -- Route multiple texts
    GET  /router/models   -- Router model metadata
"""

import asyncio
import os
import time
from typing import Any

# Prevent Ray from creating isolated worker venvs via uv.
# Must be set before Ray imports occur — Ray workers that use uv
# to create a fresh venv will fail with ModuleNotFoundError.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sentimentizer.compat  # noqa: F401
from sentimentizer import logger
from sentimentizer.predictor import SentimentPredictor
from sentimentizer.serve_base import ServiceMetrics, serve
from sentimentizer.serve_config import load_serve_config

# ---------------------------------------------------------------------------
# Load configuration (YAML defaults < env var overrides)
# ---------------------------------------------------------------------------

cfg = load_serve_config()

# ---------------------------------------------------------------------------
# FastAPI app (module-level for @serve.ingress)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentimentizer",
    version="0.210.1",
    description="Sentiment analysis and review routing API",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Centralized handler for all unhandled exceptions.

    Logs the full traceback and returns a generic 500 response
    without leaking internal details.
    """
    logger.exception("unhandled error in request")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
@serve.ingress(app)
class SentimentizerDeployment:
    """Serves sentiment analysis and review routing over HTTP.

    Uses ``serve.ingress`` with FastAPI for HTTP routing.
    ``serve.batch`` auto-batches individual ``/predict`` calls into
    efficient forward passes. Sync model inference runs via
    ``asyncio.to_thread()`` to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        self.cfg = load_serve_config()
        self._started_at = time.time()
        self._sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
        self._router_metrics = ServiceMetrics(prefix="router")

        # --- Predictor handles model loading, inference, and tokenization ---
        self.predictor = SentimentPredictor(
            model_name=self.cfg.default_model,
            router_model_path=self.cfg.router_model_path,
        )

    # ------------------------------------------------------------------
    # Auto-batched endpoints (serve.batch collects individual calls)
    # ------------------------------------------------------------------

    @serve.batch(
        max_batch_size=cfg.predict_batch_size,
        batch_wait_timeout_s=cfg.predict_batch_wait_s,
    )
    async def predict_sentiment(self, inputs: list[dict]) -> list[dict]:
        """Auto-batched sentiment prediction.

        Collects up to ``predict_batch_size`` individual ``/predict``
        requests over a ``predict_batch_wait_s`` window, then
        runs them through a single forward pass.
        """
        texts = [inp["text"] for inp in inputs]
        predictions = await asyncio.to_thread(self.predictor.predict_batch, texts)
        return predictions

    @serve.batch(
        max_batch_size=cfg.classify_batch_size,
        batch_wait_timeout_s=cfg.classify_batch_wait_s,
    )
    async def classify_route(self, inputs: list[dict]) -> list[dict]:
        """Auto-batched route classification.

        Collects up to ``classify_batch_size`` individual
        ``/router/predict`` requests, then classifies them in one batch.
        """
        texts = [inp["text"] for inp in inputs]
        results = await asyncio.to_thread(self.predictor.classify_batch, texts)
        return [r["prediction"] for r in results]

    # ------------------------------------------------------------------
    # Sentiment handlers
    # ------------------------------------------------------------------

    @app.post("/predict")
    async def predict(self, body: PredictRequest) -> dict[str, Any]:
        """POST /predict -- single text sentiment analysis (auto-batched)."""
        if len(body.text) > self.cfg.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long ({len(body.text)} chars, max {self.cfg.max_text_length})",
            )

        start = time.perf_counter()
        prediction = await self.predict_sentiment({"text": body.text})
        latency = time.perf_counter() - start

        self._sentiment_metrics.record_request(latency)
        logger.info(
            "prediction completed",
            input_length=len(body.text),
            latency_s=f"{latency:.4f}",
            model=prediction.get("model", ""),
        )
        return {
            "text": body.text,
            "prediction": prediction,
            "latency_s": round(latency, 4),
        }

    @app.post("/batch")
    async def batch(self, body: BatchRequest) -> dict[str, Any]:
        """POST /batch -- batch sentiment analysis (explicit forward pass)."""
        if len(body.texts) > self.cfg.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch too large ({len(body.texts)} items, max {self.cfg.max_batch_size})",
            )
        for i, t in enumerate(body.texts):
            if len(t) > self.cfg.max_text_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"texts[{i}] too long ({len(t)} chars, max {self.cfg.max_text_length})",
                )

        start = time.perf_counter()
        raw_predictions = await asyncio.to_thread(self.predictor.predict_batch, body.texts)
        results = [
            {"text": text, "prediction": pred}
            for text, pred in zip(body.texts, raw_predictions, strict=False)
        ]
        latency = time.perf_counter() - start

        self._sentiment_metrics.record_request(latency)
        logger.info(
            "batch prediction completed",
            batch_size=len(body.texts),
            latency_s=f"{latency:.4f}",
        )
        return {
            "results": results,
            "count": len(results),
            "latency_s": round(latency, 4),
        }

    @app.post("/tokenize")
    async def tokenize(self, body: TokenizeRequest) -> dict[str, Any]:
        """POST /tokenize -- standalone tokenization without inference."""
        if len(body.text) > self.cfg.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long ({len(body.text)} chars, max {self.cfg.max_text_length})",
            )

        result = self.predictor.tokenize(body.text)
        return result

    @app.get("/models")
    async def models(self) -> dict[str, Any]:
        """GET /models -- sentiment model metadata."""
        models_info = self.predictor.get_sentiment_model_info()
        return {"models": models_info, "default": self.predictor.model_name}

    # ------------------------------------------------------------------
    # Router handlers
    # ------------------------------------------------------------------

    @app.post("/router/predict")
    async def router_predict(self, body: PredictRequest) -> dict[str, Any]:
        """POST /router/predict -- classify a single text (auto-batched)."""
        if not self.predictor.router_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Router model not loaded: {self.predictor.router_error}",
            )

        if len(body.text) > self.cfg.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long ({len(body.text)} chars, max {self.cfg.max_text_length})",
            )

        start = time.perf_counter()
        prediction = await self.classify_route({"text": body.text})
        latency = time.perf_counter() - start

        self._router_metrics.record_request(latency)
        logger.info(
            "router prediction completed",
            input_length=len(body.text),
            latency_s=f"{latency:.4f}",
            category=prediction.get("category", ""),
        )
        return {
            "text": body.text,
            "prediction": prediction,
            "latency_s": round(latency, 4),
        }

    @app.post("/router/batch")
    async def router_batch(self, body: BatchRequest) -> dict[str, Any]:
        """POST /router/batch -- classify multiple texts into routes."""
        if not self.predictor.router_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Router model not loaded: {self.predictor.router_error}",
            )

        if len(body.texts) > self.cfg.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch too large ({len(body.texts)} items, max {self.cfg.max_batch_size})",
            )
        for i, t in enumerate(body.texts):
            if len(t) > self.cfg.max_text_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"texts[{i}] too long ({len(t)} chars, max {self.cfg.max_text_length})",
                )

        start = time.perf_counter()
        results = await asyncio.to_thread(self.predictor.classify_batch, body.texts)
        latency = time.perf_counter() - start

        self._router_metrics.record_request(latency)
        logger.info(
            "router batch completed",
            batch_size=len(body.texts),
            latency_s=f"{latency:.4f}",
        )
        return {
            "results": results,
            "latency_s": round(latency, 4),
        }

    @app.get("/router/models")
    async def router_models(self) -> dict[str, Any]:
        """GET /router/models -- router model metadata."""
        return self.predictor.get_router_model_info()

    # ------------------------------------------------------------------
    # Infrastructure handlers
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health(self) -> dict[str, Any]:
        """GET /health -- liveness / readiness probe."""
        body = {
            "status": "healthy" if self.predictor.model_loaded else "unhealthy",
            "device": self.predictor.device,
            "version": self.predictor.version,
            "uptime_s": round(time.time() - self._started_at, 1),
            "model_loaded": self.predictor.model_name,
            "router_loaded": self.predictor.router_loaded,
            "router_error": self.predictor.router_error,
        }
        if not self.predictor.model_loaded:
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get("/metrics")
    async def metrics(self) -> dict[str, Any]:
        """GET /metrics -- combined metrics for sentiment and router."""
        return {
            "sentiment": {
                "prometheus": self._sentiment_metrics.to_prometheus(),
                "request_count": self._sentiment_metrics.request_count,
                "error_count": self._sentiment_metrics.error_count,
                "avg_latency_s": self._sentiment_metrics.avg_latency_s,
            },
            "router": {
                "prometheus": self._router_metrics.to_prometheus(),
                "request_count": self._router_metrics.request_count,
                "error_count": self._router_metrics.error_count,
                "avg_latency_s": self._router_metrics.avg_latency_s,
            },
        }


# ---------------------------------------------------------------------------
# Build the Serve application
# ---------------------------------------------------------------------------

deployment = SentimentizerDeployment.bind()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Sentimentizer Serve application programmatically.

    Workers use the current Python executable directly (no isolated venv),
    which avoids the ``ModuleNotFoundError: No module named 'ray'`` issue
    where Ray's worker spawn creates a fresh venv that lacks ``ray``.
    """
    import sys

    # Prevent Ray from creating isolated worker venvs via uv.
    # Workers will use the current Python executable and inherit all
    # installed packages (including ray, torch, etc.).
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("RAY_ENABLE_RUNTIME_ENV_HOOK", "1")
    os.environ.setdefault("RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES", "")

    import ray

    ray.init(
        ignore_reinit_error=True,
        namespace="sentimentizer",
        runtime_env={
            "py_executable": sys.executable,
        },
    )
    serve.start(http_options={"host": host, "port": port})
    serve.run(deployment)
    logger.info("Sentimentizer Serve running", host=host, port=port)

    try:
        import time as _time

        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Serve")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Sentimentizer Serve")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
