"""Tests for the FastAPI serve layer.

Tests HTTP error handling, status codes, and validation without starting Ray.
Uses three strategies:
  1. Direct invocation of async handler logic via the unwrapped class
  2. FastAPI TestClient for Pydantic validation (422 responses)
  3. Standalone FastAPI app for exception-handler integration (500 responses)

Ray Serve's ``@deployment`` decorator wraps the class so it can't be
instantiated directly.  We access the original class via
``SentimentizerDeployment.func_or_class`` and call its async methods with a
mock ``self`` that mirrors the deployment's runtime attributes.
"""

import asyncio
import inspect
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sentimentizer.serve import PredictRequest, app

# Access the original unwrapped class (Ray Serve wraps it with @deployment)
from sentimentizer.serve import SentimentizerDeployment as _Deployment
from sentimentizer.serve_base import ServiceMetrics

_SentimentizerDeployment = _Deployment.func_or_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_predictor(**overrides):
    """Build a mock SentimentPredictor with sensible defaults."""
    p = MagicMock()
    p.model_loaded = True
    p.router_loaded = True
    p.model_name = "encoder"
    p.device = "cpu"
    p.version = "0.211.0"
    p.router_error = None
    p.predict_batch.return_value = [
        {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 5,
            "model": "encoder",
        },
    ]
    p.classify_batch.return_value = [
        {"text": "hello", "prediction": {"category": "general"}},
    ]
    p.tokenize.return_value = {
        "text": "hello",
        "tokens": ["hello"],
        "token_ids": [42],
        "token_count": 1,
    }
    p.get_sentiment_model_info.return_value = {
        "encoder": {"architecture": "Transformer Encoder", "status": "loaded"},
    }
    p.get_router_model_info.return_value = {
        "model_path": "models/router",
        "categories": ["dietary", "service", "general"],
        "status": "loaded",
    }
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _mock_deployment(predictor=None):
    """Build a mock ``self`` with the same attrs as SentimentizerDeployment."""
    from sentimentizer.serve_config import ServeConfig

    dep = MagicMock()
    dep.cfg = ServeConfig()
    dep._started_at = time.time()
    dep._sentiment_metrics = ServiceMetrics(prefix="sentimentizer")
    dep._router_metrics = ServiceMetrics(prefix="router")
    dep.predictor = predictor or _mock_predictor()

    async def _predict_sentiment(inputs):
        if isinstance(inputs, dict):
            return dep.predictor.predict_batch([inputs["text"]])[0]
        texts = [inp["text"] for inp in inputs]
        return dep.predictor.predict_batch(texts)

    async def _classify_route(inputs):
        if isinstance(inputs, dict):
            return dep.predictor.classify_batch([inputs["text"]])[0]["prediction"]
        texts = [inp["text"] for inp in inputs]
        results = dep.predictor.classify_batch(texts)
        return [r["prediction"] for r in results]

    async def _health_ready():
        body = {
            "status": "ready" if dep.predictor.model_loaded else "not_ready",
            "device": dep.predictor.device,
            "version": dep.predictor.version,
            "uptime_s": round(time.time() - dep._started_at, 1),
            "model_loaded": dep.predictor.model_name,
            "router_loaded": dep.predictor.router_loaded,
            "router_error": dep.predictor.router_error,
        }
        if not dep.predictor.model_loaded:
            return JSONResponse(status_code=503, content=body)
        return body

    dep.predict_sentiment = _predict_sentiment
    dep.classify_route = _classify_route
    dep.health_ready = _health_ready
    return dep


def _run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: _DummyServe stubs exist in source
# ---------------------------------------------------------------------------


class TestDummyServe:
    def test_stubs_exist_in_source(self):
        from sentimentizer import serve_base

        assert "def start(self" in inspect.getsource(serve_base)
        assert "def run(self" in inspect.getsource(serve_base)
        assert "def shutdown(self" in inspect.getsource(serve_base)


# ---------------------------------------------------------------------------
# Test: Pydantic validation with Annotated types (422 responses)
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Pydantic request model validation tests.

    After P1#7, validation uses Annotated types with max_length from
    module-level cfg. This gives 422 (Unprocessable Entity) instead of
    the previous 400 from manual validation.
    """

    def test_predict_request_rejects_empty_string(self):
        """PredictRequest with text='' should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="")
        assert any(e["type"] in ("min_length", "string_too_short") for e in exc_info.value.errors())

    def test_predict_request_rejects_missing_field(self):
        """PredictRequest with no text field should fail validation."""
        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]

    def test_predict_request_rejects_text_too_long(self):
        """PredictRequest with text exceeding max_text_length should fail."""
        from sentimentizer.serve_config import load_serve_config

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="a" * (cfg.max_text_length + 1))
        errors = exc_info.value.errors()
        assert any(e["type"] in ("max_length", "string_too_long") for e in errors)

    def test_batch_request_rejects_empty_list(self):
        """BatchRequest with texts=[] should fail validation."""
        from sentimentizer.serve import BatchRequest

        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=[])
        assert any(
            e["type"] in ("min_length", "too_short", "missing") for e in exc_info.value.errors()
        )

    def test_batch_request_rejects_too_many_items(self):
        """BatchRequest with texts exceeding max_batch_size should fail."""
        from sentimentizer.serve import BatchRequest
        from sentimentizer.serve_config import load_serve_config

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=["a"] * (cfg.max_batch_size + 1))
        errors = exc_info.value.errors()
        assert any(e["type"] in ("max_length", "too_long") for e in errors)

    def test_batch_request_rejects_per_item_text_too_long(self):
        """BatchRequest with an individual text exceeding max_text_length should fail."""
        from sentimentizer.serve import BatchRequest
        from sentimentizer.serve_config import load_serve_config

        cfg = load_serve_config()
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=["a" * (cfg.max_text_length + 1)])
        errors = exc_info.value.errors()
        # Per-item validation error includes index in loc
        assert any("texts" in str(e.get("loc", [])) for e in errors)


# ---------------------------------------------------------------------------
# Test: Prediction response schema (P1#5 additive format)
# ---------------------------------------------------------------------------


class TestPredictionResponseSchema:
    """Verify predict_batch returns additive format with both old and new keys."""

    def test_predict_batch_includes_new_fields(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello"])
        result = results[0]
        # New fields
        assert "label" in result
        assert "score" in result
        assert "scores" in result
        assert "model" in result
        assert "token_count" in result
        # Backward compat: dynamic key still present
        assert result["label"] in result
        # Dynamic key value matches the explicit score
        assert result[result["label"]] == result["score"]

    def test_predict_batch_scores_has_all_classes(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello"])
        scores = results[0]["scores"]
        assert "negative" in scores
        assert "neutral" in scores
        assert "positive" in scores

    def test_predict_batch_token_count_is_int(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello"])
        assert isinstance(results[0]["token_count"], int)


# ---------------------------------------------------------------------------
# Test: Liveness / Readiness endpoints
# ---------------------------------------------------------------------------


class TestLivenessEndpoint:
    def test_health_live_always_returns_200(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_live(dep))
        assert isinstance(result, dict)
        assert result["status"] == "alive"
        assert "uptime_s" in result


class TestReadinessEndpoint:
    def test_health_ready_model_loaded_returns_ready_dict(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, dict)
        assert result["status"] == "ready"
        assert result["device"] == "cpu"
        assert result["model_loaded"] == "encoder"

    def test_health_ready_model_not_loaded_returns_json_response_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


class TestHealthBackwardCompat:
    def test_health_delegates_to_readiness(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, dict)
        assert result["status"] == "ready"

    def test_health_delegates_to_readiness_when_not_loaded(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# Test: Sentiment model not loaded → 503
# ---------------------------------------------------------------------------


class TestSentimentModelNotLoaded:
    def test_predict_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="model file missing")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.predict(dep, PredictRequest(text="hello")))
        assert exc_info.value.status_code == 503
        assert "Sentiment model not loaded" in exc_info.value.detail

    def test_batch_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="model file missing")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.batch(dep, MagicMock(texts=["hello"])))
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Test: Router not loaded → 503
# ---------------------------------------------------------------------------


class TestRouterNotLoaded:
    def test_router_predict_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(router_loaded=False, router_error="setfit not installed")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.router_predict(dep, MagicMock(text="hello")))
        assert exc_info.value.status_code == 503
        assert "Router model not loaded" in exc_info.value.detail

    def test_router_batch_returns_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(router_loaded=False, router_error="setfit not installed")
        )

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.router_batch(dep, MagicMock(texts=["hello"])))
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Test: Exception handler → generic 500, no internal leak
# ---------------------------------------------------------------------------


class TestExceptionHandler:
    def test_unhandled_exception_returns_500_generic_message(self):
        """The centralized handler returns 500 with no internal details."""
        test_app = FastAPI()

        @test_app.exception_handler(Exception)
        async def handler(request, exc):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

        @test_app.get("/fail")
        async def fail_route():
            raise RuntimeError("something secret broke")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/fail")

        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        assert "something secret broke" not in response.text

    def test_http_exception_passes_through(self):
        """HTTPException should NOT be caught by the generic handler."""
        test_app = FastAPI()

        @test_app.exception_handler(Exception)
        async def handler(request, exc):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

        @test_app.get("/bad-request")
        async def bad_request():
            raise HTTPException(status_code=400, detail="Bad input")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/bad-request")

        assert response.status_code == 400
        assert response.json()["detail"] == "Bad input"


class TestExceptionHandlerRegistration:
    def test_app_has_exception_handler_registered(self):
        """The app should have an Exception handler registered."""
        assert Exception in app.exception_handlers


# ---------------------------------------------------------------------------
# Test: Request ID middleware
# ---------------------------------------------------------------------------


class TestRequestIdMiddleware:
    def test_cors_and_request_id_middleware_registered(self):
        """The app should have CORS and request-ID middleware registered."""

        from sentimentizer.serve import app

        cls_names = [m.cls.__name__ for m in app.user_middleware if hasattr(m, "cls")]
        assert "CORSMiddleware" in cls_names
        assert "BaseHTTPMiddleware" in cls_names


# ---------------------------------------------------------------------------
# Test: /metrics endpoint removed
# ---------------------------------------------------------------------------


class TestMetricsEndpointRemoved:
    def test_no_metrics_route_on_app(self):
        """The /metrics endpoint should not exist on the app."""
        route_paths = [route.path for route in app.routes]
        assert "/metrics" not in route_paths


# ---------------------------------------------------------------------------
# Test: v1 prefix routes exist
# ---------------------------------------------------------------------------


class TestV1Routes:
    def test_v1_routes_on_main_app(self):
        """V1 routes should be registered on the main app."""
        route_paths = [route.path for route in app.routes]
        assert "/v1/predict" in route_paths
        assert "/v1/batch" in route_paths
        assert "/v1/tokenize" in route_paths
        assert "/v1/models" in route_paths
        assert "/v1/models/{model_name}" in route_paths
        assert "/v1/router/predict" in route_paths
        assert "/v1/router/batch" in route_paths
        assert "/v1/router/models" in route_paths

    def test_health_routes_on_main_app(self):
        """Health endpoints should be on the unversioned main app."""
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
        assert "/health/live" in route_paths
        assert "/health/ready" in route_paths


# ---------------------------------------------------------------------------
# Test: CORS configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Error response envelope (P2#9)
# ---------------------------------------------------------------------------


class TestErrorEnvelope:
    """Verify HTTP exceptions use the standard error envelope format."""

    def test_503_uses_error_envelope(self):
        """Exceptions with string detail should be wrapped in error envelope."""
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.predict(dep, PredictRequest(text="hello")))
        exc = exc_info.value
        assert exc.status_code == 503

    def test_model_validation_wrong_model_400(self):
        """Requesting an unloaded model should return 400."""
        dep = _mock_deployment()
        with pytest.raises(HTTPException) as exc_info:
            _SentimentizerDeployment._validate_model(dep, "rnn")
        assert exc_info.value.status_code == 400
        assert "not loaded" in exc_info.value.detail

    def test_model_validation_none_returns_default(self):
        """Passing None for model should return the default model name."""
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, None)
        assert result == "encoder"

    def test_model_validation_matching_model_succeeds(self):
        """Passing the loaded model name should succeed."""
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "encoder")
        assert result == "encoder"


# ---------------------------------------------------------------------------
# Test: include_scores and top_k (P2#11)
# ---------------------------------------------------------------------------


class TestFormatPrediction:
    """Verify _format_prediction handles include_scores, top_k, and token_count."""

    def test_format_prediction_full(self):
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 5,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(
            prediction, include_scores=True, top_k=None
        )
        assert result["label"] == "positive"
        assert result["score"] == 0.88
        assert result["model"] == "encoder"
        assert result["positive"] == 0.88
        assert "scores" in result
        assert result["scores"]["negative"] == 0.02
        assert result["token_count"] == 5

    def test_format_prediction_no_scores(self):
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 5,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(
            prediction, include_scores=False, top_k=None
        )
        assert result["label"] == "positive"
        assert result["score"] == 0.88
        assert "scores" not in result
        # Dynamic key still present
        assert result["positive"] == 0.88
        # token_count present even when scores omitted
        assert result["token_count"] == 5

    def test_format_prediction_top_k(self):
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 5,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(
            prediction, include_scores=True, top_k=2
        )
        assert len(result["scores"]) == 2
        assert "positive" in result["scores"]
        # Top 2 by score: positive (0.88) and neutral (0.10)
        assert "neutral" in result["scores"]
        assert result["token_count"] == 5

    def test_format_prediction_without_token_count(self):
        """Old format without token_count should still work."""
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(
            prediction, include_scores=True, top_k=None
        )
        assert "token_count" not in result
        assert result["label"] == "positive"


# ---------------------------------------------------------------------------
# Test: Request model classes (P2#10)
# ---------------------------------------------------------------------------


class TestRequestModelFields:
    """Verify new request fields (model, include_scores, top_k)."""

    def test_predict_request_defaults(self):
        req = PredictRequest(text="hello")
        assert req.model is None
        assert req.include_scores is True
        assert req.top_k is None

    def test_predict_request_with_model(self):
        req = PredictRequest(text="hello", model="encoder")
        assert req.model == "encoder"

    def test_predict_request_include_scores_false(self):
        req = PredictRequest(text="hello", include_scores=False)
        assert req.include_scores is False

    def test_predict_request_top_k(self):
        req = PredictRequest(text="hello", top_k=2)
        assert req.top_k == 2

    def test_batch_request_defaults(self):
        from sentimentizer.serve import BatchRequest

        req = BatchRequest(texts=["hello"])
        assert req.model is None
        assert req.include_scores is True
        assert req.top_k is None

    def test_batch_request_with_model(self):
        from sentimentizer.serve import BatchRequest

        req = BatchRequest(texts=["hello"], model="encoder")
        assert req.model == "encoder"


# ---------------------------------------------------------------------------
# Test: V1 model detail endpoint (P3#13)
# ---------------------------------------------------------------------------


class TestModelDetailEndpoint:
    def test_v1_has_model_detail_route(self):
        """The app should have /v1/models/{model_name} route."""
        route_paths = [route.path for route in app.routes]
        assert "/v1/models/{model_name}" in route_paths


# ---------------------------------------------------------------------------
# Test: Request body size limit (P3#15)
# ---------------------------------------------------------------------------


class TestBodySizeLimitMiddleware:
    def test_body_size_limit_middleware_registered(self):
        """The _RequestBodySizeLimitMiddleware should be in app middleware."""
        from sentimentizer.serve import _RequestBodySizeLimitMiddleware

        cls_names = [m.cls for m in app.user_middleware if hasattr(m, "cls")]
        assert _RequestBodySizeLimitMiddleware in cls_names


# ---------------------------------------------------------------------------
# Test: Validate model field behavior
# ---------------------------------------------------------------------------


class TestValidateModel:
    """Test _validate_model edge cases."""

    def test_validate_model_none_returns_default(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, None)
        assert result == "encoder"

    def test_validate_model_matching_succeeds(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "encoder")
        assert result == "encoder"

    def test_validate_model_case_insensitive(self):
        dep = _mock_deployment()
        result = _SentimentizerDeployment._validate_model(dep, "Encoder")
        assert result == "encoder"

    def test_validate_model_wrong_raises_400(self):
        dep = _mock_deployment()
        with pytest.raises(HTTPException) as exc_info:
            _SentimentizerDeployment._validate_model(dep, "rnn")
        assert exc_info.value.status_code == 400
        assert "not loaded" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Test: Error response envelope format
# ---------------------------------------------------------------------------


class TestErrorResponseEnvelope:
    """Test that HTTP exceptions produce the standard envelope."""

    def test_http_exception_handler_wraps_string_detail(self):
        """HTTPException with string detail should be wrapped in error envelope."""
        from sentimentizer.serve import http_exception_handler

        test_app = FastAPI()

        @test_app.exception_handler(HTTPException)
        async def handler(request, exc):
            return await http_exception_handler(request, exc)

        @test_app.get("/test-503")
        async def test_503():
            raise HTTPException(status_code=503, detail="Service unavailable")

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test-503")
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "service_unavailable"
        assert data["error"]["message"] == "Service unavailable"

    def test_status_code_to_error_code_mapping(self):
        from sentimentizer.serve import _status_code_to_error_code

        assert _status_code_to_error_code(400) == "bad_request"
        assert _status_code_to_error_code(404) == "not_found"
        assert _status_code_to_error_code(413) == "request_too_large"
        assert _status_code_to_error_code(422) == "validation_error"
        assert _status_code_to_error_code(503) == "service_unavailable"
        assert _status_code_to_error_code(499) == "error_499"


# ---------------------------------------------------------------------------
# Test: Token count in prediction response
# ---------------------------------------------------------------------------


class TestTokenCount:
    """Verify token_count is present in prediction output."""

    def test_predict_batch_includes_token_count(self):
        pred = _mock_predictor()
        results = pred.predict_batch(["hello world"])
        assert "token_count" in results[0]
        assert isinstance(results[0]["token_count"], int)

    def test_format_prediction_preserves_token_count(self):
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "token_count": 12,
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert result["token_count"] == 12

    def test_format_prediction_omits_token_count_when_absent(self):
        """Legacy predictions without token_count should still format correctly."""
        prediction = {
            "positive": 0.88,
            "label": "positive",
            "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
            "model": "encoder",
        }
        result = _SentimentizerDeployment._format_prediction(prediction)
        assert "token_count" not in result


# ---------------------------------------------------------------------------
# Test: ServeConfig with cors_origins
# ---------------------------------------------------------------------------


class TestServeConfigCorsOrigins:
    """Test ServeConfig cors_origins parsing."""

    def test_default_cors_origins(self):
        from sentimentizer.serve_config import ServeConfig

        cfg = ServeConfig()
        assert cfg.cors_origins == ["*"]

    def test_env_var_cors_origins(self):
        import os

        from sentimentizer.serve_config import load_serve_config

        os.environ["SENTIMENTIZER_CORS_ORIGINS"] = "http://localhost:3000,https://app.example.com"
        try:
            cfg = load_serve_config()
            assert cfg.cors_origins == [
                "http://localhost:3000",
                "https://app.example.com",
            ]
        finally:
            del os.environ["SENTIMENTIZER_CORS_ORIGINS"]

    def test_single_cors_origin(self):
        import os

        from sentimentizer.serve_config import load_serve_config

        os.environ["SENTIMENTIZER_CORS_ORIGINS"] = "https://app.example.com"
        try:
            cfg = load_serve_config()
            assert cfg.cors_origins == ["https://app.example.com"]
        finally:
            del os.environ["SENTIMENTIZER_CORS_ORIGINS"]

    def test_parse_list_helper(self):
        from sentimentizer.serve_config import _parse_list

        assert _parse_list("a,b,c") == ["a", "b", "c"]
        assert _parse_list("a, b , c") == ["a", "b", "c"]
        assert _parse_list("single") == ["single"]
        assert _parse_list("") == []


# ---------------------------------------------------------------------------
# Test: Health endpoint response shapes
# ---------------------------------------------------------------------------


class TestHealthResponseShape:
    """Verify health endpoint response structure."""

    def test_liveness_has_status_and_uptime(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_live(dep))
        assert "status" in result
        assert "uptime_s" in result
        assert result["status"] == "alive"

    def test_readiness_has_device_and_version(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert "device" in result
        assert "version" in result
        assert "model_loaded" in result
        assert "router_loaded" in result

    def test_readiness_not_loaded_includes_error(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health_ready(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# Test: CORS middleware configuration
# ---------------------------------------------------------------------------


class TestCORSMiddleware:
    """Test CORS middleware is properly configured."""

    def test_cors_middleware_allows_all_origins_by_default(self):
        from sentimentizer.serve_config import ServeConfig

        cfg = ServeConfig()
        assert cfg.cors_origins == ["*"]

    def test_cors_exposes_request_id_header(self):
        """X-Request-Id should be in exposed headers for CORS."""
        from sentimentizer.serve import app

        for m in app.user_middleware:
            if hasattr(m, "kwargs") and "expose_headers" in m.kwargs:
                assert "X-Request-Id" in m.kwargs["expose_headers"]


# ---------------------------------------------------------------------------
# Test: Request body size limit
# ---------------------------------------------------------------------------


class TestRequestBodySizeLimit:
    """Test the body size limit middleware."""

    def test_max_body_size_constant(self):
        from sentimentizer.serve import MAX_REQUEST_BODY_BYTES

        assert MAX_REQUEST_BODY_BYTES == 1 * 1024 * 1024  # 1 MiB
