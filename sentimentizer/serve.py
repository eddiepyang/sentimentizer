"""Serve the Sentimentizer pipeline via Ray Serve.

Provides a unified REST API for:
  - Sentiment analysis (RNN, Encoder, Decoder models)
  - Review routing (Dietary, Service, General categories)

Usage:
    ray serve run sentimentizer.serve:app

Endpoints:
  Sentiment analysis:
    POST /predict         — Classify a single text
    POST /batch           — Classify multiple texts
    POST /tokenize        — Tokenize text without inference
    GET  /models          — Sentiment model metadata
    GET  /health          — Health check
    GET  /metrics          — Request metrics

  Router (review categorization):
    POST /router/predict  — Route a single text
    POST /router/batch     — Route multiple texts
    GET  /router/models    — Router model metadata
"""

import os
import time
from pathlib import Path
from typing import Any

import torch
from starlette.requests import Request
from starlette.responses import JSONResponse

# Apply transformers compatibility shim BEFORE importing setfit
import sentimentizer.compat  # noqa: F401
from sentimentizer import logger
from sentimentizer.config import auto_detect_device
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.decoder import get_trained_model as get_decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.encoder import get_trained_model as get_encoder
from sentimentizer.models.rnn import RNN
from sentimentizer.models.rnn import get_trained_model as get_rnn
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
# Model registries
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "rnn": {
        "loader": get_rnn,
        "class": RNN,
        "architecture": "Bidirectional LSTM",
        "embedding_dim": 100,
        "hidden_size": 256,
        "num_layers": 2,
    },
    "encoder": {
        "loader": get_encoder,
        "class": Encoder,
        "architecture": "Transformer Encoder (CLS token)",
        "embedding_dim": 100,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
    },
    "decoder": {
        "loader": get_decoder,
        "class": Decoder,
        "architecture": "Encoder-Decoder Transformer (cross-attention)",
        "embedding_dim": 100,
        "d_model": 256,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
    },
}

DEFAULT_MODEL = "rnn"

# ---------------------------------------------------------------------------
# Metrics (separate prefixes for sentiment vs router)
# ---------------------------------------------------------------------------

sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
router_metrics = ServiceMetrics(prefix="router")

# ---------------------------------------------------------------------------
# Health state
# ---------------------------------------------------------------------------

_health_state: dict = {
    "loaded": False,
    "device": "unknown",
    "models": list(MODEL_REGISTRY.keys()),
    "router_loaded": False,
    "router_model_path": None,
}

# ---------------------------------------------------------------------------
# Combined deployment
# ---------------------------------------------------------------------------


@serve.deployment(
    route_prefix="/",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 5,
        "metrics_interval_s": 10,
    },
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
        # --- Sentiment models ---
        self.device: str = auto_detect_device()
        self.tokenizer: Tokenizer = get_trained_tokenizer()
        self.models: dict[str, torch.nn.Module] = {}
        for model_name, registry_entry in MODEL_REGISTRY.items():
            loader = registry_entry["loader"]
            model = loader(device=self.device)
            model.to(self.device)
            model.eval()
            self.models[model_name] = model

        # --- Router model ---
        if router_model_path is None:
            router_model_path = os.environ.get("ROUTER_MODEL_PATH", "models/router")
        self.router_model_path = Path(router_model_path)

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

        # --- Update health state ---
        _health_state["loaded"] = True
        _health_state["device"] = self.device
        _health_state["router_loaded"] = True
        _health_state["router_model_path"] = str(self.router_model_path)

    # ---- Sentiment prediction logic ----------------------------------------

    def _get_model(self, model_name: str | None) -> tuple[torch.nn.Module, str]:
        """Resolve the sentiment model, falling back to DEFAULT_MODEL."""
        name = (model_name or DEFAULT_MODEL).lower().strip()
        if name not in self.models:
            name = DEFAULT_MODEL
        return self.models[name], name

    def _predict_sentiment(self, text: str, model_name: str | None = None) -> dict:
        """Run sentiment analysis on a single text."""
        model, resolved_name = self._get_model(model_name)
        processed_input = self.tokenizer.tokenize_text(text)
        score = model.predict(processed_input).item()
        return {
            "model": resolved_name,
            "sentiment_score": score,
            "label": "positive" if score > 0.5 else "negative",
        }

    # ---- Router prediction logic --------------------------------------------

    def _classify_single(self, text: str) -> dict:
        """Classify a single text into a route category."""
        predictions = self.router.predict([text])
        label = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        if isinstance(label, (int, float)):
            label = RouteLabels.label_names().get(int(label), str(label))
        return {"category": label, "categories": RouteLabels.label_names()}

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts into route categories."""
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

            if model_name and model_name.lower().strip() not in MODEL_REGISTRY:
                available = list(MODEL_REGISTRY.keys())
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
                score=prediction["sentiment_score"],
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            sentiment_metrics.record_request(latency, error=True)
            logger.exception("sentiment prediction failed")
            return build_error_response(f"Internal error: {exc}", status_code=500)

    async def _handle_batch(self, http_request: Request) -> JSONResponse:
        """POST /batch — batch sentiment analysis."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            texts = json_input.get("texts", [])
            model_name = json_input.get("model")

            if not isinstance(texts, list) or len(texts) == 0:
                return build_error_response("No texts provided or texts is not a non-empty list")

            if model_name and model_name.lower().strip() not in MODEL_REGISTRY:
                available = list(MODEL_REGISTRY.keys())
                return build_error_response(f"Unknown model: {model_name}. Available: {available}")

            results = [
                {"text": t, "prediction": self._predict_sentiment(t, model_name)} for t in texts
            ]
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
        """GET /models — sentiment model metadata."""
        models_info = {}
        for name, registry_entry in MODEL_REGISTRY.items():
            model = self.models[name]
            param_count = sum(p.numel() for p in model.parameters())
            info: dict[str, Any] = {
                "architecture": registry_entry["architecture"],
                "device": self.device,
                "max_sequence_length": self.tokenizer.cfg.max_len,
                "embedding_dim": registry_entry["embedding_dim"],
                "parameters": param_count,
                "status": "loaded",
            }
            if name == "rnn":
                info["hidden_size"] = registry_entry["hidden_size"]
                info["num_layers"] = registry_entry["num_layers"]
            elif name == "encoder":
                info["d_model"] = registry_entry["d_model"]
                info["n_heads"] = registry_entry["n_heads"]
                info["n_layers"] = registry_entry["n_layers"]
            elif name == "decoder":
                info["d_model"] = registry_entry["d_model"]
                info["n_heads"] = registry_entry["n_heads"]
                info["n_encoder_layers"] = registry_entry["n_encoder_layers"]
                info["n_decoder_layers"] = registry_entry["n_decoder_layers"]
            models_info[name] = info

        return JSONResponse({"models": models_info, "default": DEFAULT_MODEL})

    # ---- Router handlers ----------------------------------------------------

    async def _handle_router_predict(self, http_request: Request) -> JSONResponse:
        """POST /router/predict — classify a single text into a route."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            text = json_input.get("text", "")

            if not text or not isinstance(text, str):
                return build_error_response("No text provided or text is not a string")

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
            json_input = await http_request.json()
            texts = json_input.get("texts", [])

            if not isinstance(texts, list) or len(texts) == 0:
                return build_error_response("No texts provided or texts is not a non-empty list")

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
        return JSONResponse(
            {
                "model_path": str(self.router_model_path),
                "categories": RouteLabels.label_names(),
                "status": "loaded",
            }
        )

    # ---- Shared infrastructure handlers -------------------------------------

    async def _handle_health(self, http_request: Request) -> JSONResponse:
        """GET /health — liveness / readiness probe."""
        return build_health_response(**_health_state)

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
