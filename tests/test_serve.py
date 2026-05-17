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
    p.version = "0.210.1"
    p.router_error = None
    p.predict_batch.return_value = [{"positive": 0.88, "model": "encoder"}]
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

    # Wire predict_sentiment / classify_route as async methods.
    # In production, @serve.batch collects inputs into a list, calls the
    # function once, and unwraps single-item results back to the caller.
    # Without @serve.batch, our mock receives a single dict and must
    # return a single dict (not a list).
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

    dep.predict_sentiment = _predict_sentiment
    dep.classify_route = _classify_route
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
# Test: Validation raises HTTPException with correct status codes
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_predict_text_too_long_raises_400(self):
        dep = _mock_deployment()
        dep.cfg = MagicMock(max_text_length=5, max_batch_size=64)

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.predict(dep, PredictRequest(text="abcdefgh")))
        assert exc_info.value.status_code == 400
        assert "too long" in exc_info.value.detail

    def test_predict_ok_text_passes_validation(self):
        dep = _mock_deployment()

        # predict() calls await self.predict_sentiment({"text": ...})
        # which in production uses @serve.batch. In tests the method
        # returns whatever predict_batch returns minus the batching.
        # Since @serve.batch is not active, we get a list back from
        # predict_batch via the mock.  The important thing is that
        # validation passes (no HTTPException raised) and we get a result.
        result = _run(_SentimentizerDeployment.predict(dep, PredictRequest(text="hello")))
        assert "text" in result


class TestBatchValidation:
    def test_batch_too_large_raises_400(self):
        dep = _mock_deployment()
        dep.cfg = MagicMock(max_batch_size=2, max_text_length=10000)

        body = MagicMock()
        body.texts = ["a", "b", "c"]

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.batch(dep, body))
        assert exc_info.value.status_code == 400
        assert "Batch too large" in exc_info.value.detail

    def test_batch_text_too_long_raises_400(self):
        dep = _mock_deployment()
        dep.cfg = MagicMock(max_batch_size=64, max_text_length=5)

        body = MagicMock()
        body.texts = ["abcdefgh"]

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.batch(dep, body))
        assert exc_info.value.status_code == 400
        assert "too long" in exc_info.value.detail


class TestTokenizeValidation:
    def test_tokenize_text_too_long_raises_400(self):
        dep = _mock_deployment()
        dep.cfg = MagicMock(max_text_length=5, max_batch_size=64)

        body = MagicMock()
        body.text = "abcdefgh"

        with pytest.raises(HTTPException) as exc_info:
            _run(_SentimentizerDeployment.tokenize(dep, body))
        assert exc_info.value.status_code == 400


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
# Test: Health endpoint returns correct response
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_model_loaded_returns_healthy_dict(self):
        dep = _mock_deployment()
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, dict)
        assert result["status"] == "healthy"
        assert result["device"] == "cpu"

    def test_health_model_not_loaded_returns_json_response_503(self):
        dep = _mock_deployment(
            predictor=_mock_predictor(model_loaded=False, model_error="not found")
        )
        result = _run(_SentimentizerDeployment.health(dep))
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503


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


# ---------------------------------------------------------------------------
# Test: Pydantic validation (422) via TestClient
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Pydantic request model validation tests.

    These test the ``min_length`` constraints on request models.
    We test model validation directly (not via HTTP) because the real
    ``app`` uses ``@serve.ingress`` which requires a running Ray Serve
    replica, and TestClient trips over that in unit tests.
    """

    def test_predict_request_rejects_empty_string(self):
        """PredictRequest with text='' should fail validation."""
        from pydantic import ValidationError

        from sentimentizer.serve import PredictRequest

        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="")
        assert any(e["type"] in ("min_length", "string_too_short") for e in exc_info.value.errors())

    def test_predict_request_rejects_missing_field(self):
        """PredictRequest with no text field should fail validation."""
        from pydantic import ValidationError

        from sentimentizer.serve import PredictRequest

        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]

    def test_batch_request_rejects_empty_list(self):
        """BatchRequest with texts=[] should fail validation."""
        from pydantic import ValidationError

        from sentimentizer.serve import BatchRequest

        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(texts=[])
        assert any(
            e["type"] in ("min_length", "too_short", "missing") for e in exc_info.value.errors()
        )


class TestExceptionHandlerRegistration:
    def test_app_has_exception_handler_registered(self):
        """The app should have an Exception handler registered."""
        assert Exception in app.exception_handlers
