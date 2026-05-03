import threading
import time
from typing import Any

import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from sentimentizer import logger
from sentimentizer.config import auto_detect_device
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.decoder import get_trained_model as get_decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.encoder import get_trained_model as get_encoder
from sentimentizer.models.rnn import RNN
from sentimentizer.models.rnn import get_trained_model as get_rnn
from sentimentizer.tokenizer import Tokenizer, get_trained_tokenizer, regex_tokenize, text_sequencer

# ---------------------------------------------------------------------------
# Model registry — maps model names to their loader functions and classes
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
        "n_decoder_layers": 4,
    },
}

DEFAULT_MODEL = "rnn"

# ---------------------------------------------------------------------------
# Metrics helpers (lightweight Prometheus-compatible counters/histograms)
# ---------------------------------------------------------------------------


class Metrics:
    """Simple in-memory metrics collector with thread-safe counters."""

    def __init__(self) -> None:
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
                "# HELP sentimentizer_request_total Total requests processed",
                "# TYPE sentimentizer_request_total counter",
                f"sentimentizer_request_total {self.request_count}",
                "# HELP sentimentizer_error_total Total errors",
                "# TYPE sentimentizer_error_total counter",
                f"sentimentizer_error_total {self.error_count}",
                "# HELP sentimentizer_latency_seconds_total Cumulative latency in seconds",
                "# TYPE sentimentizer_latency_seconds_total counter",
                f"sentimentizer_latency_seconds_total {self.total_latency_s:.6f}",
            ]
            return "\n".join(lines) + "\n"


metrics = Metrics()


# ---------------------------------------------------------------------------
# Shared health state — set by deployment __init__, read by /health route
# ---------------------------------------------------------------------------

_health_state: dict = {
    "loaded": False,
    "device": "unknown",
    "models": list(MODEL_REGISTRY.keys()),
}


# ---------------------------------------------------------------------------
# Model deployment — handles /, /predict, /batch, /tokenize, /models
# ---------------------------------------------------------------------------


@serve.deployment(
    route_prefix="/",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 5,
        "metrics_interval_s": 10,
    },
    max_ongoing_requests=10,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
class SentimentDeployment:
    """Deployment server for all sentiment models (RNN, Encoder, Decoder)."""

    def __init__(self) -> None:
        self.device: str = auto_detect_device()
        self.tokenizer: Tokenizer = get_trained_tokenizer()

        # Load all models at startup
        self.models: dict[str, torch.nn.Module] = {}
        for model_name, registry_entry in MODEL_REGISTRY.items():
            loader = registry_entry["loader"]
            model = loader(device=self.device)
            model.to(self.device)
            model.eval()
            self.models[model_name] = model

        # Update shared health state
        _health_state["loaded"] = True
        _health_state["device"] = self.device

    def _get_model(self, model_name: str | None) -> tuple[torch.nn.Module, str]:
        """Resolve the model to use, returning (model, resolved_name).

        Falls back to DEFAULT_MODEL if model_name is None or unknown.
        """
        name = (model_name or DEFAULT_MODEL).lower().strip()
        if name not in self.models:
            name = DEFAULT_MODEL
        return self.models[name], name

    # ---- Core prediction logic ------------------------------------------------

    def _predict_single(self, text: str, model_name: str | None = None) -> dict:
        """Run inference on a single text string."""
        model, resolved_name = self._get_model(model_name)
        processed_input = self.tokenizer.tokenize_text(text)
        prediction_tensor = model.predict(processed_input)
        score = prediction_tensor.item()
        return {
            "text": text,
            "model": resolved_name,
            "sentiment_score": score,
            "prediction": "positive" if score > 0.5 else "negative",
        }

    # ---- Route handlers -------------------------------------------------------

    async def __call__(self, http_request: Request) -> JSONResponse:
        """Dispatch requests based on URL path."""
        path = http_request.url.path.rstrip("/")

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
            return JSONResponse({"error": f"Unknown path: {path}"}, status_code=404)

    async def _handle_predict(self, http_request: Request) -> JSONResponse:
        """POST / or /predict — single text prediction."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            text = json_input.get("text", "")
            model_name = json_input.get("model")

            if not text or not isinstance(text, str):
                return JSONResponse(
                    {"error": "No text provided or text is not a string"}, status_code=400
                )

            # Validate model name if provided
            if model_name and model_name.lower().strip() not in MODEL_REGISTRY:
                available = list(MODEL_REGISTRY.keys())
                return JSONResponse(
                    {"error": f"Unknown model: {model_name}. Available: {available}"},
                    status_code=400,
                )

            result = self._predict_single(text, model_name)
            latency = time.perf_counter() - start
            metrics.record_request(latency)
            logger.info(  # type: ignore[call-arg]
                "prediction completed",
                model=result["model"],
                input_length=len(text),
                prediction=result["prediction"],
                score=result["sentiment_score"],
                latency_s=f"{latency:.4f}",
            )
            return JSONResponse(result)
        except Exception as exc:
            latency = time.perf_counter() - start
            metrics.record_request(latency, error=True)
            logger.exception("prediction failed")
            return JSONResponse({"error": f"Internal error: {exc}"}, status_code=500)

    async def _handle_batch(self, http_request: Request) -> JSONResponse:
        """POST /batch — batch prediction for multiple texts."""
        start = time.perf_counter()
        try:
            json_input = await http_request.json()
            texts = json_input.get("texts", [])
            model_name = json_input.get("model")

            if not isinstance(texts, list) or len(texts) == 0:
                return JSONResponse(
                    {"error": "No texts provided or texts is not a non-empty list"},
                    status_code=400,
                )

            # Validate model name if provided
            if model_name and model_name.lower().strip() not in MODEL_REGISTRY:
                available = list(MODEL_REGISTRY.keys())
                return JSONResponse(
                    {"error": f"Unknown model: {model_name}. Available: {available}"},
                    status_code=400,
                )

            results = [self._predict_single(t, model_name) for t in texts]
            latency = time.perf_counter() - start
            metrics.record_request(latency)
            logger.info(  # type: ignore[call-arg]
                "batch prediction completed",
                model=results[0]["model"] if results else "unknown",
                batch_size=len(texts),
                latency_s=f"{latency:.4f}",
            )
            return JSONResponse({"results": results, "count": len(results)})
        except Exception as exc:
            latency = time.perf_counter() - start
            metrics.record_request(latency, error=True)
            logger.exception("batch prediction failed")
            return JSONResponse({"error": f"Internal error: {exc}"}, status_code=500)

    async def _handle_tokenize(self, http_request: Request) -> JSONResponse:
        """POST /tokenize — standalone tokenization without inference."""
        try:
            json_input = await http_request.json()
            text = json_input.get("text", "")

            if not text or not isinstance(text, str):
                return JSONResponse(
                    {"error": "No text provided or text is not a string"}, status_code=400
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
            return JSONResponse({"error": f"Internal error: {exc}"}, status_code=500)

    async def _handle_models(self, http_request: Request) -> JSONResponse:
        """GET /models — metadata about all available models."""
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
            # Add model-specific metadata
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

        return JSONResponse(
            {
                "models": models_info,
                "default": DEFAULT_MODEL,
            }
        )

    async def _handle_health(self, http_request: Request) -> JSONResponse:
        """GET /health — K8s liveness / readiness probe target."""
        if _health_state["loaded"]:
            return JSONResponse({"status": "healthy", **_health_state})
        return JSONResponse({"status": "unhealthy", **_health_state}, status_code=503)

    async def _handle_metrics(self, http_request: Request) -> JSONResponse:
        """GET /metrics — Prometheus-compatible metrics."""
        return JSONResponse(
            {
                "prometheus": metrics.to_prometheus(),
                "request_count": metrics.request_count,
                "error_count": metrics.error_count,
                "avg_latency_s": (
                    metrics.total_latency_s / metrics.request_count if metrics.request_count else 0
                ),
            }
        )


# ---------------------------------------------------------------------------
# Build the Serve application
# ---------------------------------------------------------------------------

app = SentimentDeployment.bind()
