import time

import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from sentimentizer import logger
from sentimentizer.models.rnn import RNN, get_trained_model
from sentimentizer.tokenizer import Tokenizer, get_trained_tokenizer


# ---------------------------------------------------------------------------
# Metrics helpers (lightweight Prometheus-compatible counters/histograms)
# ---------------------------------------------------------------------------


class Metrics:
    """Simple in-memory metrics collector."""

    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_s: float = 0.0

    def record_request(self, latency_s: float, error: bool = False) -> None:
        self.request_count += 1
        self.total_latency_s += latency_s
        if error:
            self.error_count += 1

    def to_prometheus(self) -> str:
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

_health_state: dict = {"loaded": False, "device": "unknown", "model": "RNN"}


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
    """deployment server for the RNN sentiment model"""

    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: RNN = get_trained_model(batch_size=1, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer: Tokenizer = get_trained_tokenizer()

        # Update shared health state
        _health_state["loaded"] = True
        _health_state["device"] = self.device

    # ---- Core prediction logic ------------------------------------------------

    def _predict_single(self, text: str) -> dict:
        """Run inference on a single text string."""
        processed_input = self.tokenizer.tokenize_text(text)
        prediction_tensor = self.model.predict(processed_input)
        score = prediction_tensor.item()
        return {
            "text": text,
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

            if not text or not isinstance(text, str):
                return JSONResponse(
                    {"error": "No text provided or text is not a string"}, status_code=400
                )

            result = self._predict_single(text)
            latency = time.perf_counter() - start
            metrics.record_request(latency)
            logger.info(  # type: ignore[call-arg]
                "prediction completed",
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

            if not isinstance(texts, list) or len(texts) == 0:
                return JSONResponse(
                    {"error": "No texts provided or texts is not a non-empty list"},
                    status_code=400,
                )

            results = [self._predict_single(t) for t in texts]
            latency = time.perf_counter() - start
            metrics.record_request(latency)
            logger.info(  # type: ignore[call-arg]
                "batch prediction completed",
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

            from sentimentizer.tokenizer import regex_tokenize, text_sequencer

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
        """GET /models — metadata about the loaded model."""
        return JSONResponse(
            {
                "model": "RNN",
                "architecture": "LSTM",
                "device": self.device,
                "max_sequence_length": 200,
                "embedding_dim": 100,
                "status": "loaded",
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
                    metrics.total_latency_s / metrics.request_count
                    if metrics.request_count
                    else 0
                ),
            }
        )


# ---------------------------------------------------------------------------
# Build the Serve application
# ---------------------------------------------------------------------------

app = SentimentDeployment.bind()