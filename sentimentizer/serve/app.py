"""Serve the Sentimentizer pipeline via Ray Serve.

Provides a unified REST API for:
  - Sentiment analysis (encoder model by default)
  - Review routing (Dietary, Service, General categories)

Uses ``serve.batch`` to auto-batch individual ``/v1/predict`` and
``/v1/router/predict`` requests into efficient forward passes, and
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
    POST /v1/predict           -- Classify a single text (auto-batched)
    POST /v1/batch             -- Classify multiple texts (single forward pass)
    POST /v1/tokenize          -- Tokenize text without inference
    GET  /v1/models             -- Sentiment model metadata (all models)
    GET  /v1/models/{name}     -- Single model metadata

  Router (review categorization):
    POST /v1/router/predict   -- Route a single text (auto-batched)
    POST /v1/router/batch     -- Route multiple texts
    GET  /v1/router/models    -- Router model metadata

  Infrastructure (unversioned):
    GET  /health               -- Backward-compatible alias for /health/ready
    GET  /health/live          -- Liveness probe (always 200)
    GET  /health/ready         -- Readiness probe (503 if model not loaded)
"""

import asyncio
import os
import time
import uuid
from typing import Any

# Prevent Ray from creating isolated worker venvs via uv.
# Must be set before Ray imports occur — Ray workers that use uv
# to create a fresh venv will fail with ModuleNotFoundError.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from collections.abc import Callable, Coroutine

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

import sentimentizer.compat  # noqa: F401
from sentimentizer import logger
from sentimentizer.predictor import _MODEL_CONFIGS, SentimentPredictor
from sentimentizer.serve.base import ServiceMetrics, serve
from sentimentizer.serve.config import load_serve_config
from sentimentizer.serve.models import (
    BatchRequest,
    BatchResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    ModelDetailResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    RouterBatchResponse,
    RouterModelsResponse,
    RouterPredictResponse,
    TokenizeRequest,
    TokenizeResponse,
)

# ---------------------------------------------------------------------------
# Load configuration (YAML defaults < env var overrides)
# ---------------------------------------------------------------------------

cfg = load_serve_config()

# ---------------------------------------------------------------------------
# FastAPI app (module-level for @serve.ingress)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentimentizer",
    version="0.211.0",
    description="Sentiment analysis and review routing API",
)

# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------
# Order matters! Starlette processes middleware in LIFO order:
#   - Last added = outermost (first request in, last response out)
# Required order (outermost first):
#   1. CORS — must be outermost to handle preflight OPTIONS before auth
#   2. Request-ID — adds trace IDs to all requests including CORS preflight
#   3. Body size limit — reject oversized payloads early
# ---------------------------------------------------------------------------

MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024  # 1 MiB defense-in-depth limit


class _RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with body exceeding MAX_REQUEST_BODY_BYTES.

    Defense-in-depth: the K8s ingress already enforces
    ``proxy-body-size: "1m"``, but this middleware catches requests
    that bypass the ingress (e.g., port-forward, node port).
    """

    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[
            [StarletteRequest], Coroutine[StarletteRequest, None, StarletteResponse]
        ],
    ) -> StarletteResponse:
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "request_too_large",
                            "message": (f"Request body exceeds " f"{MAX_REQUEST_BODY_BYTES} bytes"),
                        }
                    },
                )
        return await call_next(request)


app.add_middleware(_RequestBodySizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


@app.middleware("http")
async def request_id_middleware(
    request: StarletteRequest,
    call_next: Callable[[StarletteRequest], Coroutine[StarletteRequest, None, StarletteResponse]],
) -> StarletteResponse:
    """Add X-Request-Id to every request/response for distributed tracing."""
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Centralized handler for all unhandled exceptions.

    Logs the full traceback and returns a generic 500 response
    without leaking internal details. Uses the standard error envelope.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled error in request", request_id=request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Internal server error",
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap HTTPException in the standard error envelope.

    Converts FastAPI/Starlette's default ``{"detail": "..."}`` format
    into the structured ``{"error": {"code": ..., "message": ...}}``
    envelope. Validation errors (422) from Pydantic are left as-is
    since they already have structured ``detail`` arrays.
    """
    detail = exc.detail
    if isinstance(detail, str):
        error_code = _status_code_to_error_code(exc.status_code)
        content = {
            "error": {
                "code": error_code,
                "message": detail,
            }
        }
    else:
        content = (
            {"error": detail} if isinstance(detail, dict) else {"error": {"message": str(detail)}}
        )
    return JSONResponse(status_code=exc.status_code, content=content)


def _status_code_to_error_code(status_code: int) -> str:
    """Map common HTTP status codes to machine-readable error codes."""
    mapping = {
        400: "bad_request",
        404: "not_found",
        413: "request_too_large",
        422: "validation_error",
        503: "service_unavailable",
    }
    return mapping.get(status_code, f"error_{status_code}")


# ---------------------------------------------------------------------------
# Pydantic request models (use module-level cfg for validation limits)
# ---------------------------------------------------------------------------


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
    ``serve.batch`` auto-batches individual ``/v1/predict`` calls into
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

    def _validate_model(self, model_name: str | None) -> str:
        """Validate requested model matches loaded model.

        Returns the validated model name. Raises HTTPException(400)
        if the requested model doesn't match the loaded model.
        """
        if model_name is None:
            return self.predictor.model_name
        if model_name.lower().strip() != self.predictor.model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_name}' is not loaded. "
                    f"Available model: '{self.predictor.model_name}'"
                ),
            )
        return self.predictor.model_name

    @staticmethod
    def _format_prediction(
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a prediction dict for the response.

        The prediction dict has the additive v1 format:
        ``{label: score, "label": label, "score": score,
        "scores": {...}, "token_count": N, "model": model}``.
        Returns label, score, token_count, and model.
        """
        result: dict[str, Any] = {
            "label": prediction["label"],
            "score": prediction["score"],
            "model": prediction["model"],
        }
        if "token_count" in prediction:
            result["token_count"] = prediction["token_count"]
        return result

    # ------------------------------------------------------------------
    # Auto-batched endpoints (serve.batch collects individual calls)
    # ------------------------------------------------------------------

    @serve.batch(
        max_batch_size=cfg.predict_batch_size,
        batch_wait_timeout_s=cfg.predict_batch_wait_s,
    )
    async def predict_sentiment(self, inputs: list[dict]) -> list[dict]:
        """Auto-batched sentiment prediction.

        Collects up to ``predict_batch_size`` individual ``/v1/predict``
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
        ``/v1/router/predict`` requests, then classifies them in one batch.
        """
        texts = [inp["text"] for inp in inputs]
        results = await asyncio.to_thread(self.predictor.classify_batch, texts)
        return [r["prediction"] for r in results]

    # ------------------------------------------------------------------
    # Sentiment handlers (v1 prefix)
    # ------------------------------------------------------------------

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(self, body: PredictRequest) -> dict[str, Any]:
        """POST /v1/predict -- single text sentiment analysis (auto-batched)."""
        if not self.predictor.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Sentiment model not loaded: {self.predictor.model_error}",
            )
        self._validate_model(body.model)

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
        formatted = self._format_prediction(prediction)
        return {
            "prediction": formatted,
            "latency_s": round(latency, 4),
        }

    @app.post("/v1/batch", response_model=BatchResponse)
    async def batch(self, body: BatchRequest) -> dict[str, Any]:
        """POST /v1/batch -- batch sentiment analysis (explicit forward pass)."""
        if not self.predictor.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Sentiment model not loaded: {self.predictor.model_error}",
            )
        self._validate_model(body.model)

        start = time.perf_counter()
        raw_predictions = await asyncio.to_thread(self.predictor.predict_batch, body.texts)
        results = [
            {
                "prediction": self._format_prediction(pred),
            }
            for pred in raw_predictions
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

    @app.post("/v1/tokenize", response_model=TokenizeResponse)
    async def tokenize(self, body: TokenizeRequest) -> dict[str, Any]:
        """POST /v1/tokenize -- standalone tokenization without inference."""
        result = await asyncio.to_thread(self.predictor.tokenize, body.text)
        return result

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models(self) -> dict[str, Any]:
        """GET /v1/models -- sentiment model metadata (all models)."""
        models_info = self.predictor.get_sentiment_model_info()
        return {"models": models_info, "default": self.predictor.model_name}

    @app.get("/v1/models/{model_name}", response_model=ModelDetailResponse)
    async def model_detail(
        self,
        model_name: str = Path(..., description="Model name: rnn, encoder, or decoder"),
    ) -> dict[str, Any]:
        """GET /v1/models/{model_name} -- single model metadata."""
        if model_name not in _MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Available: {list(_MODEL_CONFIGS.keys())}",
            )
        info = self.predictor.get_sentiment_model_info()
        model_info = info.get(model_name)
        if model_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in model info.",
            )
        return {"model": model_name, "info": model_info}

    # ------------------------------------------------------------------
    # Router handlers (v1 prefix)
    # ------------------------------------------------------------------

    @app.post("/v1/router/predict", response_model=RouterPredictResponse)
    async def router_predict(self, body: PredictRequest) -> dict[str, Any]:
        """POST /v1/router/predict -- classify a single text (auto-batched)."""
        if not self.predictor.router_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Router model not loaded: {self.predictor.router_error}",
            )

        start = time.perf_counter()
        prediction = await self.classify_route({"text": body.text})
        latency = time.perf_counter() - start

        self._router_metrics.record_request(latency)
        logger.info(
            "router prediction completed",
            input_length=len(body.text),
            latency_s=f"{latency:.4f}",
            category=prediction.get("label", ""),
        )
        return {
            "prediction": prediction,
            "latency_s": round(latency, 4),
        }

    @app.post("/v1/router/batch", response_model=RouterBatchResponse)
    async def router_batch(self, body: BatchRequest) -> dict[str, Any]:
        """POST /v1/router/batch -- classify multiple texts into routes."""
        if not self.predictor.router_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Router model not loaded: {self.predictor.router_error}",
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
            "count": len(results),
            "latency_s": round(latency, 4),
        }

    @app.get("/v1/router/models", response_model=RouterModelsResponse)
    async def router_models(self) -> dict[str, Any]:
        """GET /v1/router/models -- router model metadata."""
        return self.predictor.get_router_model_info()

    # ------------------------------------------------------------------
    # Infrastructure handlers (unversioned)
    # ------------------------------------------------------------------

    @app.get("/health/live", response_model=HealthLiveResponse)
    async def health_live(self) -> dict[str, Any]:
        """GET /health/live -- liveness probe (always returns 200)."""
        return {
            "status": "alive",
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    @app.get("/health/ready", response_model=HealthReadyResponse)
    async def health_ready(self) -> dict[str, Any] | JSONResponse:
        """GET /health/ready -- readiness probe (503 if model not loaded)."""
        body = {
            "status": "ready" if self.predictor.model_loaded else "not_ready",
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

    @app.get("/health", response_model=HealthReadyResponse)
    async def health(self) -> dict[str, Any] | JSONResponse:
        """GET /health -- backward-compatible alias for /health/ready."""
        return await self.health_ready()


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
