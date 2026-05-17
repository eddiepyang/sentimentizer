"""Serve the Sentimentizer pipeline via Ray Serve.

Provides a unified REST API for:
  - Sentiment analysis (RNN, Encoder, Decoder models)
  - Review routing (Dietary, Service, General categories)

Usage:
    ray serve run sentimentizer.serve:app

Endpoints:
  Sentiment analysis:
    POST /predict         — Classify a single text
    POST /batch           — Classify multiple texts (single forward pass)
    POST /tokenize        — Tokenize text without inference
    GET  /models          — Sentiment model metadata
    GET  /health          — Health check
    GET  /metrics          — Request metrics

  Router (review categorization):
    POST /router/predict  — Route a single text
    POST /router/batch     — Route multiple texts
    GET  /router/models    — Router model metadata
"""

import contextlib
import dataclasses
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from starlette.requests import Request
from starlette.responses import JSONResponse

# Apply transformers compatibility shim BEFORE importing setfit
import sentimentizer.compat  # noqa: F401
from sentimentizer import logger
from sentimentizer.config import (
    DecoderConfig,
    EmbeddingsConfig,
    EncoderConfig,
    RNNConfig,
    auto_detect_device,
)
from sentimentizer.models.base import BaseSentimentModel, get_trained_model
from sentimentizer.router.config import RouteLabels, SetFitConfig
from sentimentizer.router.train_router import _load_setfit_model
from sentimentizer.serve_base import (
    ServiceMetrics,
    build_batch_response,
    build_error_response,
    build_health_response,
    build_predict_response,
    serve,
)
from sentimentizer.tokenizer import Tokenizer, get_trained_tokenizer, regex_tokenize, text_sequencer

# ---------------------------------------------------------------------------
# Request limits (configurable via env vars)
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE: int = int(os.environ.get("SENTIMENTIZER_MAX_BATCH_SIZE", "64"))
MAX_TEXT_LENGTH: int = int(os.environ.get("SENTIMENTIZER_MAX_TEXT_LENGTH", "10000"))

# ---------------------------------------------------------------------------
# Model config registry — metadata derived from config dataclasses
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "rnn": {
        "architecture": "Bidirectional LSTM",
        "config_class": RNNConfig,
    },
    "encoder": {
        "architecture": "Transformer Encoder (CLS token)",
        "config_class": EncoderConfig,
    },
    "decoder": {
        "architecture": "Encoder-Decoder Transformer (cross-attention)",
        "config_class": DecoderConfig,
    },
}

DEFAULT_MODEL = "rnn"

# ---------------------------------------------------------------------------
# Enabled models — configurable via SENTIMENTIZER_MODELS env var
# ---------------------------------------------------------------------------

_enabled_raw = os.environ.get("SENTIMENTIZER_MODELS", ",".join(MODEL_CONFIGS))
ENABLED_MODELS: list[str] = [
    m.strip().lower() for m in _enabled_raw.split(",") if m.strip().lower() in MODEL_CONFIGS
]
if not ENABLED_MODELS:
    ENABLED_MODELS = [DEFAULT_MODEL]

# ---------------------------------------------------------------------------
# Metrics (separate prefixes for sentiment vs router)
# ---------------------------------------------------------------------------

sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
router_metrics = ServiceMetrics(prefix="router")

# ---------------------------------------------------------------------------
# Combined deployment
# ---------------------------------------------------------------------------


@serve.deployment(
    route_prefix="/",
    # Single Ray replica per pod — K8s HPA handles pod-level scaling.
    # Removes the dual-autoscaler conflict with the previous autoscaling_config.
    num_replicas=1,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class SentimentizerDeployment:
    """Unified deployment serving both sentiment analysis and review routing.

    Routes:
        /predict, /batch, /tokenize, /models   -> Sentiment analysis
        /router/predict, /router/batch          -> Review categorization
        /health, /metrics                         -> Shared infrastructure
    """

    def __init__(self, router_model_path: str | None = None) -> None:
        self._started_at = time.time()

        # --- Sentiment models (only ENABLED_MODELS are loaded) ---
        self.device: str = auto_detect_device()
        self.tokenizer: Tokenizer = get_trained_tokenizer()
        self.models: dict[str, torch.nn.Module] = {}
        self._model_errors: dict[str, str] = {}

        for model_name in ENABLED_MODELS:
            try:
                model = get_trained_model(model_name, device=self.device)
                model.tokenizer = self.tokenizer  # required for predict_text()
                model.to(self.device)
                model.eval()
                self.models[model_name] = model
                logger.info(f"Loaded sentiment model: {model_name}")
            except Exception:
                logger.exception(f"Failed to load model {model_name}")
                self._model_errors[model_name] = "load failed"

        if not self.models:
            raise RuntimeError(
                f"No sentiment models could be loaded. Attempted: {ENABLED_MODELS}. "
                f"Errors: {self._model_errors}"
            )

        # --- Router model (graceful degradation if unavailable) ---
        if router_model_path is None:
            router_model_path = os.environ.get("ROUTER_MODEL_PATH", "models/router")
        self.router_model_path = Path(router_model_path)
        self.router = None
        self._router_error: str | None = None

        try:
            from setfit import SetFitModel

            if self.router_model_path.exists():
                logger.info(f"Loading router model from {self.router_model_path}")
                self.router = SetFitModel.from_pretrained(str(self.router_model_path))
            else:
                logger.info("Loading router base model with classification head")
                config = SetFitConfig()
                self.router = _load_setfit_model(
                    config.base_model, num_classes=RouteLabels.num_classes()
                )
        except Exception as exc:
            logger.exception("Failed to load router model — router endpoints disabled")
            self._router_error = str(exc)

        # --- Version ---
        try:
            import importlib.metadata

            self._version = importlib.metadata.version("sentimentizer")
        except Exception:
            self._version = "unknown"

    def __del__(self) -> None:
        """Log clean shutdown for observability."""
        with contextlib.suppress(Exception):
            logger.info(
                "SentimentizerDeployment shutting down",
                models=list(getattr(self, "models", {}).keys()),
            )

    # ---- Sentiment prediction logic ----------------------------------------

    def _get_model(self, model_name: str | None) -> tuple[torch.nn.Module, str]:
        """Resolve the sentiment model, falling back to first available."""
        name = (model_name or DEFAULT_MODEL).lower().strip()
        if name not in self.models:
            name = next(iter(self.models))
        return self.models[name], name

    def _predict_sentiment(self, text: str, model_name: str | None = None) -> dict:
        """Run sentiment analysis on a single text."""
        model, resolved_name = self._get_model(model_name)
        scores = model.predict_text(text)
        label = max(scores, key=scores.get)
        return {
            "model": resolved_name,
            "label": label,
            "scores": scores,
        }

    def _predict_sentiment_batch(
        self, texts: list[str], model_name: str | None = None
    ) -> list[dict]:
        """Run batched sentiment analysis — single forward pass for all texts.

        Tokenizes all texts, stacks into a single (B, seq_len) array,
        and runs one forward pass instead of N individual passes.
        """
        model, resolved_name = self._get_model(model_name)
        token_arrays = [model.tokenizer.tokenize_text(t) for t in texts]
        batch = np.concatenate(token_arrays, axis=0)  # (B, seq_len)
        probs = model.predict(batch)  # (B, num_classes) tensor

        label_names = BaseSentimentModel.LABEL_NAMES
        results = []
        for i, text in enumerate(texts):
            scores = {lbl: probs[i, j].item() for j, lbl in enumerate(label_names)}
            label = max(scores, key=scores.get)
            results.append(
                {
                    "text": text,
                    "prediction": {"model": resolved_name, "label": label, "scores": scores},
                }
            )
        return results

    # ---- Router prediction logic --------------------------------------------

    def _classify_single(self, text: str) -> dict:
        """Classify a single text into a route category."""
        if self.router is None:
            raise RuntimeError(
                f"Router model is not loaded: {self._router_error or 'unknown error'}"
            )
        predictions = self.router.predict([text])
        label = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        if isinstance(label, (int, float)):
            label = RouteLabels.label_names().get(int(label), str(label))
        return {"category": label, "categories": RouteLabels.label_names()}

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts into route categories."""
        if self.router is None:
            raise RuntimeError(
                f"Router model is not loaded: {self._router_error or 'unknown error'}"
            )
        predictions = self.router.predict(texts)
        results = []
        for text, pred in zip(texts, predictions, strict=False):
            label = (
                pred
                if isinstance(pred, str)
                else RouteLabels.label_names().get(int(pred), str(pred))
            )
            results.append({"text": text, "prediction": {"category": label}})
        return results

    # ---- Request dispatch ---------------------------------------------------

    async def __call__(self, http_request: Request) -> JSONResponse:
        """Dispatch requests to sentiment or router handlers."""
        path = http_request.url.path.rstrip("/")

        # Router routes
        router_path = path.removeprefix("/router") if path.startswith("/router") else None
        if router_path is not None:
            if router_path in ("", "/predict"):
                return await self._handle_router_predict(http_request)
            elif router_path == "/batch":
                return await self._handle_router_batch(http_request)
            elif router_path == "/models":
                return await self._handle_router_models(http_request)
            else:
                return build_error_response(f"Unknown router path: {path}", status_code=404)

        # Sentiment routes
        if path in ("", "/predict"):
            return await self._handle_predict(http_request)
        elif path == "/batch":
            return await self._handle_batch(http_request)
        elif path == "/tokenize":
            return await self._handle_tokenize(http_request)
        elif path == "/models":
            return await self._handle_models(http_request)
        elif path == "/health":
            return await self._handle_health(http_request)
        elif path == "/metrics":
            return await self._handle_metrics(http_request)
        else:
            return build_error_response(f"Unknown path: {path}", status_code=404)

    # ---- Sentiment handlers -------------------------------------------------

    async def _handle_predict(self, http_request: Request) -> JSONResponse:
        """POST /predict — single text sentiment analysis."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            text = json_input.get("text", "")
            model_name = json_input.get("model")

            if not text or not isinstance(text, str):
                return build_error_response("No text provided or text is not a string")

            if len(text) > MAX_TEXT_LENGTH:
                return build_error_response(
                    f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
                )

            if model_name and model_name.lower().strip() not in self.models:
                available = list(self.models.keys())
                return build_error_response(f"Unknown model: {model_name}. Available: {available}")

            prediction = self._predict_sentiment(text, model_name)
            latency = time.perf_counter() - start
            return build_predict_response(
                text=text,
                prediction=prediction,
                latency_s=latency,
                metrics=sentiment_metrics,
                model=prediction["model"],
                label=prediction["label"],
                scores=prediction["scores"],
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            sentiment_metrics.record_request(latency, error=True)
            logger.exception("sentiment prediction failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_batch(self, http_request: Request) -> JSONResponse:
        """POST /batch — batch sentiment analysis (single forward pass)."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            texts = json_input.get("texts", [])
            model_name = json_input.get("model")

            if not isinstance(texts, list) or len(texts) == 0:
                return build_error_response("No texts provided or texts is not a non-empty list")

            if len(texts) > MAX_BATCH_SIZE:
                return build_error_response(
                    f"Batch too large ({len(texts)} items, max {MAX_BATCH_SIZE})"
                )

            # Validate each text before running inference
            for i, t in enumerate(texts):
                if not isinstance(t, str):
                    return build_error_response(f"texts[{i}] is not a string")
                if len(t) > MAX_TEXT_LENGTH:
                    return build_error_response(
                        f"texts[{i}] too long ({len(t)} chars, max {MAX_TEXT_LENGTH})"
                    )

            if model_name and model_name.lower().strip() not in self.models:
                available = list(self.models.keys())
                return build_error_response(f"Unknown model: {model_name}. Available: {available}")

            results = self._predict_sentiment_batch(texts, model_name)
            latency = time.perf_counter() - start
            return build_batch_response(
                results=results,
                latency_s=latency,
                metrics=sentiment_metrics,
                model=model_name or DEFAULT_MODEL,
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            sentiment_metrics.record_request(latency, error=True)
            logger.exception("sentiment batch failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_tokenize(self, http_request: Request) -> JSONResponse:
        """POST /tokenize — standalone tokenization without inference."""
        try:
            json_input = await http_request.json()
            text = json_input.get("text", "")

            if not text or not isinstance(text, str):
                return build_error_response("No text provided or text is not a string")

            if len(text) > MAX_TEXT_LENGTH:
                return build_error_response(
                    f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
                )

            tokens = regex_tokenize(text)
            token_ids = text_sequencer(
                self.tokenizer.dictionary, tokens, self.tokenizer.cfg.max_len
            )

            return JSONResponse(
                {
                    "text": text,
                    "tokens": tokens,
                    "token_ids": token_ids.tolist(),
                    "token_count": len(tokens),
                }
            )
        except Exception as exc:
            logger.exception("tokenization failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_models(self, http_request: Request) -> JSONResponse:
        """GET /models — sentiment model metadata (derived from config dataclasses)."""
        emb_dim = EmbeddingsConfig.emb_length
        models_info: dict[str, Any] = {}

        for name, model in self.models.items():
            config_entry = MODEL_CONFIGS[name]
            cfg = config_entry["config_class"]()
            cfg_dict = dataclasses.asdict(cfg)
            param_count = sum(p.numel() for p in model.parameters())

            models_info[name] = {
                "architecture": config_entry["architecture"],
                "device": self.device,
                "max_sequence_length": self.tokenizer.cfg.max_len,
                "embedding_dim": emb_dim,
                "parameters": param_count,
                "status": "loaded",
                **cfg_dict,
            }

        # Report models that failed to load
        for name, err in self._model_errors.items():
            models_info[name] = {"status": "error", "error": err}

        return JSONResponse({"models": models_info, "default": DEFAULT_MODEL})

    # ---- Router handlers ----------------------------------------------------

    async def _handle_router_predict(self, http_request: Request) -> JSONResponse:
        """POST /router/predict — classify a single text into a route."""
        start = time.perf_counter()
        try:
            if self.router is None:
                return build_error_response(
                    f"Router model not loaded: {self._router_error}", status_code=503
                )

            json_input = await http_request.json()
            text = json_input.get("text", "")

            if not text or not isinstance(text, str):
                return build_error_response("No text provided or text is not a string")

            if len(text) > MAX_TEXT_LENGTH:
                return build_error_response(
                    f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
                )

            prediction = self._classify_single(text)
            latency = time.perf_counter() - start
            return build_predict_response(
                text=text,
                prediction=prediction,
                latency_s=latency,
                metrics=router_metrics,
                log_name="router prediction",
                category=prediction["category"],
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            router_metrics.record_request(latency, error=True)
            logger.exception("router prediction failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_router_batch(self, http_request: Request) -> JSONResponse:
        """POST /router/batch — classify multiple texts into routes."""
        start = time.perf_counter()
        try:
            if self.router is None:
                return build_error_response(
                    f"Router model not loaded: {self._router_error}", status_code=503
                )

            json_input = await http_request.json()
            texts = json_input.get("texts", [])

            if not isinstance(texts, list) or len(texts) == 0:
                return build_error_response("No texts provided or texts is not a non-empty list")

            if len(texts) > MAX_BATCH_SIZE:
                return build_error_response(
                    f"Batch too large ({len(texts)} items, max {MAX_BATCH_SIZE})"
                )

            results = self._classify_batch(texts)
            latency = time.perf_counter() - start
            return build_batch_response(
                results=results,
                latency_s=latency,
                metrics=router_metrics,
                log_name="router batch",
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            router_metrics.record_request(latency, error=True)
            logger.exception("router batch failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_router_models(self, http_request: Request) -> JSONResponse:
        """GET /router/models — router model metadata."""
        status = "loaded" if self.router is not None else "error"
        resp: dict[str, Any] = {
            "model_path": str(self.router_model_path),
            "categories": RouteLabels.label_names(),
            "status": status,
        }
        if self._router_error:
            resp["error"] = self._router_error
        return JSONResponse(resp)

    # ---- Shared infrastructure handlers -------------------------------------

    async def _handle_health(self, http_request: Request) -> JSONResponse:
        """GET /health — liveness / readiness probe."""
        loaded = len(self.models) > 0
        return build_health_response(
            loaded=loaded,
            device=self.device,
            version=self._version,
            uptime_s=round(time.time() - self._started_at, 1),
            models_loaded=list(self.models.keys()),
            models_failed=self._model_errors,
            router_loaded=self.router is not None,
            router_error=self._router_error,
        )

    async def _handle_metrics(self, http_request: Request) -> JSONResponse:
        """GET /metrics — combined metrics for sentiment and router."""
        return JSONResponse(
            {
                "sentiment": {
                    "prometheus": sentiment_metrics.to_prometheus(),
                    "request_count": sentiment_metrics.request_count,
                    "error_count": sentiment_metrics.error_count,
                    "avg_latency_s": sentiment_metrics.avg_latency_s,
                },
                "router": {
                    "prometheus": router_metrics.to_prometheus(),
                    "request_count": router_metrics.request_count,
                    "error_count": router_metrics.error_count,
                    "avg_latency_s": router_metrics.avg_latency_s,
                },
            }
        )


# ---------------------------------------------------------------------------
# Build the Serve application
# ---------------------------------------------------------------------------

app = SentimentizerDeployment.bind()
