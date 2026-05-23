# Serve API Best Practices — Implementation Plan

Risk/gap annotations are marked with **[RISK]**, **[GAP]**, or **[SAFE]**.

---

## P0 — Do now

### 1. Split `/health` into liveness + readiness

**Problem**: Single `/health` conflates liveness (process alive) and readiness (can serve traffic). If a pod takes 30s to load a model, K8s liveness probe fails → restart loop.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Add `GET /health/live` — always returns 200 `{"status": "alive", "uptime_s": ...}`
  - Rename current `/health` logic to `GET /health/ready` — 503 if model not loaded
  - Keep `GET /health` as backward-compatible alias that delegates to readiness (same response body, **not** a redirect — K8s probes don't follow redirects by default)
  - Keep `/docs` and `/redoc` at root (unversioned) — they're operational tooling, not business API
- `k8s/deployment.yaml`:
  - `livenessProbe.httpGet.path` → `/health/live`
  - `readinessProbe.httpGet.path` → `/health/ready`
  - `startupProbe.httpGet.path` → `/health/ready` (startup checks readiness)
- `k8s/service.yaml`, `k8s/ingress.yaml`:
  - No changes (service routes all traffic, ingress matches `/`)
- `tests/test_serve.py`:
  - Add `TestLivenessEndpoint` (always 200, "alive")
  - Rename `TestHealthEndpoint` → `TestReadinessEndpoint`
  - Add `TestHealthBackwardCompat` (delegates to readiness)

**[SAFE]** No downstream callers to worry about — K8s probes are the only consumer and we control their config.

**[RISK]** If `startupProbe` uses `/health/ready` and model loading takes longer than `failureThreshold * periodSeconds` (currently 300s), the pod will be killed before it starts. Currently `/health` already returns 503 when model is not loaded, so `startupProbe` is already checking readiness. The `startupProbe` exists specifically to allow slow startup — if the startup probe succeeds, K8s stops checking it and switches to liveness/readiness. **No behavioral change for startup.** The key improvement is that `livenessProbe` now uses `/health/live` which always succeeds, preventing restart loops during model loading.

**[REVIEW NOTE]** Add `uptime_s` to `/health/live` response for operational visibility (already in `/health` today). Keep `/health` as backward-compatible alias that delegates to `/health/ready` (same response body, not a redirect — K8s probes don't follow redirects).

---

### 2. Add `/v1/` URL prefix

**Problem**: No versioning means any breaking response change breaks all clients. Every major ML API uses URL-prefix versioning.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Use explicit `/v1/` prefixes on the root `app` directly (e.g., `@app.post("/v1/predict")`). Do NOT use `APIRouter`.
  - Keep `/health`, `/health/live`, `/health/ready` at root (unversioned)
  - Update module docstring endpoint list
- `workflows/cli.py`:
  - Update docstring listing endpoint paths (in `serve_cmd()` function)
- `tests/test_serve.py`:
  - Update any hardcoded paths

**[CRITICAL DISCOVERY] `APIRouter` incompatibility.** We discovered that `APIRouter` sub-apps do not work properly with Ray Serve deployments (specifically regarding route registration and/or `@serve.batch` interaction). Therefore, we must register all `/v1/` routes directly on the root `FastAPI` app instance using explicit path prefixes.

**[SAFE]** Verified that using explicit `/v1/` prefixes directly on the root app works seamlessly with `@serve.ingress` and `@serve.batch`.

**[RISK] Breaking change for existing clients.** Any client currently calling `POST /predict` will get 404 after this change. Mitigations:
  - **Option A (recommended)**: Add a deprecation shim — keep old paths as aliases that redirect (307) to `/v1/...` versions. Remove after a transition period.
  - **Option B**: Cut over with no backward compat — coordinate with consumers (this is an internal service, not a public API, so the blast radius is controlled).

**[REVIEW NOTE]** Given this is an internal service with controlled blast radius, Option B (clean cutover) is simpler and avoids maintaining redirect shims. The `/v1/` prefix makes the API version explicit, and any internal clients can be updated in the same deploy cycle. If backward compat is needed, Option A is documented above.

**[GAP]** The plan originally proposed renaming endpoints too (`/predict` → `/v1/sentiments`). This mixes two changes (versioning + rename) that should be separate PRs. Renaming URLs means updating all K8s ingress rules, internal client scripts, and documentation simultaneously. **Recommendation**: P0 is just the `/v1/` prefix. Keep current path names under `/v1/` (e.g., `/v1/predict`, `/v1/batch`). Path renaming is a separate P2 change.

**[REVIEW NOTE]** Document the deprecation timeline for the dynamic key in P1 #5 schema change: the intermediate format (both old and new keys) will persist until `/v2/`. Clients must migrate within one major version cycle.

**Revised endpoint map for P0**:

| Old | New |
|-----|-----|
| POST `/predict` | POST `/v1/predict` |
| POST `/batch` | POST `/v1/batch` |
| POST `/tokenize` | POST `/v1/tokenize` |
| GET `/models` | GET `/v1/models` |
| POST `/router/predict` | POST `/v1/router/predict` |
| POST `/router/batch` | POST `/v1/router/batch` |
| GET `/router/models` | GET `/v1/router/models` |
| GET `/health` | GET `/health` (unversioned) |
| GET `/health/live` | GET `/health/live` (unversioned) |
| GET `/health/ready` | GET `/health/ready` (unversioned) |
| GET `/docs` | GET `/docs` (unversioned) |
| GET `/redoc` | GET `/redoc` (unversioned) |

**Also in this PR**: Add `"count": len(results)` to the `/v1/router/batch` response to match the sentiment `/v1/batch` response shape. Trivial 1-liner, no reason to defer.

---

### 3. Remove `/metrics` from user API

**Problem**: Exposes internal operational data on user-facing port. Prometheus format embedded as string in JSON (useless for scraping). Standalone exporter on 8081 already serves proper Prometheus exposition format.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Delete `GET /metrics` handler and `_sentiment_metrics`/`_router_metrics` from the deployment
  - Remove from module docstring endpoint list
- `workflows/cli.py`:
  - Remove `/metrics` from docstring (in `serve_cmd()` function)
- `tests/test_serve.py`:
  - Remove any metrics endpoint tests (none currently exist)
- `sentimentizer/serve/base.py`:
  - `ServiceMetrics` class stays — it's used by the standalone exporter

**[RISK] Breaking change if any client relies on /metrics.** The JSON `/metrics` endpoint is not used by Prometheus (which scrapes port 8081 in the standalone exporter). But someone might have built a dashboard or script that reads it. **Mitigation**: Check with consumers before removing, or keep as deprecated with a log warning for one release cycle.

**[GAP]** The plan didn't account for `_sentiment_metrics` and `_router_metrics` being used in the handlers. Currently, `predict()`, `batch()`, `router_predict()`, and `router_batch()` all call `self._sentiment_metrics.record_request(latency)` or `self._router_metrics.record_request(latency)`. If we remove `/metrics`, we still need per-request timing for observability. **Two options**:
  - **Option A (recommended)**: Keep `ServiceMetrics` in the deployment for internal observability and structured logging, but remove the JSON endpoint. The `record_request()` calls continue to work, and we can expose these metrics via Prometheus gauges on port 8081 instead.
  - **Option B**: Remove both the endpoint and the metrics tracking from the deployment entirely. Simpler but loses per-replica request counts and latencies.

**[GAP]** The standalone exporter (`sentimentizer/exporter.py`, port 8081) reads from per-model-type JSON files written by the training pipeline. It does **not** scrape the `/metrics` endpoint on port 8000. So removing `/metrics` from the serve deployment does not break the standalone exporter. But the per-replica `ServiceMetrics` data (request_count, error_count, avg_latency_s) is currently only exposed via `/metrics`. If we delete the endpoint, this data becomes unreachable. **Recommendation**: Option A — keep `ServiceMetrics`, delete the HTTP endpoint, and add Prometheus gauge pushes from the deployment to the exporter (or expose on a separate metrics port, deferrable to P3).

**[GAP] Dead code after removal.** `ServiceMetrics.to_prometheus()` generates Prometheus exposition format text, but it's only ever consumed as a string embedded in the JSON `/metrics` response. After removing the endpoint, `to_prometheus()` becomes dead code. **Recommendation**: Keep it for now — it will be needed by the P3 Prometheus push. Add a `# TODO(P3): used by future Prometheus push` comment.

**[REVIEW NOTE]** Option A is the clear choice. Remove the `/metrics` HTTP endpoint completely (no deprecation period — this is not a Prometheus endpoint and the data is useless as JSON-embedded strings). Keep `ServiceMetrics` for internal observability and structured logging. Add the TODO comment on `to_prometheus()`. The per-replica data (request_count, error_count, avg_latency_s) will be exposed via the standalone exporter on port 8081 in P3.

---

## P1 — Do soon

### 4. Add request IDs

**Problem**: No way to trace individual requests through logs.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Add middleware that reads `X-Request-Id` from request headers, generates UUID if absent, attaches to response headers and `request.state`
  - Log request ID alongside predictions
  - Add `id` field (request ID) to prediction responses (coordinate with P1 #5 schema change to avoid two separate response format updates)

**[RISK] Middleware execution order.** FastAPI/Starlette processes middleware in LIFO order — the last middleware added is the outermost. If `CORSMiddleware` (P1 #6) is added later, it will wrap the request-ID middleware. This is correct (CORS needs to be outermost to handle preflight), but we need to document the middleware registration order as a convention. **Recommendation**: Add a comment block above the middleware registrations stating the required order.

**[REVIEW NOTE]** Required middleware registration order (outermost first): (1) CORS, (2) Request ID. This means CORS middleware is added LAST in code (LIFO execution) and request-ID is added second-to-last. Document this convention in a comment block above `app.add_middleware()` calls.

**[RISK] `@serve.batch` and request IDs.** When `predict()` calls `await self.predict_sentiment({"text": body.text})`, the `@serve.batch` decorator collects multiple calls and runs them as one batch. The request ID from the original request is available in `predict()` but does not automatically propagate into the batch function. For tracing, we need to either:
  - **Option A**: Log the request ID only in `predict()` (pre-batch), not in `predict_sentiment()`. This is usually sufficient since the latency measurement and result construction happen in `predict()`.
  - **Option B**: Pass the request ID through the batch dict (`{"text": ..., "request_id": ...}`) so `predict_sentiment` can log per-batch-item. More complete but more complex.
  **Recommendation**: Option A for simplicity. The batch function is an internal implementation detail — the external request ID trace point is the handler.

**[SAFE]** UUID generation is cheap (no I/O, no coordination). No performance concern.

---

### 5. Fix `prediction` response schema

**Problem**: Current `{"positive": 0.88, "model": "encoder"}` uses winning label as dynamic key. Clients must iterate keys to find label, other class scores are lost.

**Target schema**:
```json
{
  "label": "positive",
  "score": 0.88,
  "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88},
  "model": "encoder"
}
```

**Changes**:

- `sentimentizer/predictor.py`:
  - Change `predict_batch()` return format from `{label: score, "model": model_name}` to `{"label": label, "score": score, "scores": all_3, "model": model_name}`
- `sentimentizer/serve/app.py`:
  - Update response construction in handlers that consume `predict_batch` output
- `AGENTS.md`:
  - Update "Option B format" description
- `tests/test_serve.py`:
  - Update mock predictor return values

**[RISK] Breaking change — the single biggest risk in this plan.** This changes the wire format that all clients consume. Specifically:
  - `predict_batch()` return format is used by: (1) `serve.py` handlers, (2) `agent/diagnose_model.py` validation (but `diagnose_model.py` uses `model.predict_text()` not `predict_batch()`, so **[SAFE]**).
  - `predict()` (single text wrapper) calls `predict_batch([text])[0]` and returns the first element. If `predict_batch` output changes, `predict()` return changes automatically.
  - **`serve.py` handlers**: The `predict()` handler does `prediction.get("model", "")` and `prediction = await self.predict_sentiment(...)`. After the schema change, `prediction` will have `{"label", "score", "scores", "model"}` — the `.get("model")` still works. But the response embeds `prediction` directly, so the client-facing response shape changes.
  - **`/batch` handler**: Constructs `{"text": text, "prediction": pred}` where `pred` is the raw `predict_batch` output. Same shape change.

**[RISK] `predict_text()` on `BaseSentimentModel` is a different API surface.** Per AGENTS.md: "`predict_text()` on `BaseSentimentModel` still returns all 3 scores: `{"negative": 0.05, "neutral": 0.12, "positive": 0.83}`. This is a different API surface used by `diagnose_model.py` and `hf.py`." That API is **not** affected by this change — it's on the model class, not the predictor. **[SAFE]**

**[GAP] `predictor.py` module docstring is already aspirational.** The docstring shows `{"model": "encoder", "label": "positive", "scores": {"negative": 0.02, ...}}` (the target format), but the actual code in `predict_batch()` returns Option B `{label: score, "model": model_name}`. Fix the docstring as part of this change.

**[GAP] Versioning dependency — P1 #5 MUST come after P0 #2.** This schema change requires URL versioning to be in place first. **Decision: implement as a non-breaking additive change under `/v1/`.** Add `label`, `score`, `scores` as new fields while keeping the dynamic key for backward compatibility:

```json
{
  "positive": 0.88,
  "model": "encoder",
  "label": "positive",
  "score": 0.88,
  "scores": {"negative": 0.02, "neutral": 0.10, "positive": 0.88}
}
```

Clients can migrate to the new fields at their own pace. The dynamic key will be removed in a future `/v2/` prefix, avoiding two forced client migrations.

**[REVIEW NOTE]** The deprecation timeline: the dynamic key (e.g., `"positive": 0.88`) is deprecated as of `/v1/` launch. It will be removed when `/v2/` is introduced. Clients should migrate to `label`, `score`, `scores` within one major version cycle.

---

### 6. Add CORS middleware

**Problem**: No CORS configured. Any browser-based client is blocked.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Add `CORSMiddleware` to app
- `sentimentizer/serve/config.py`:
  - Add `cors_origins: list[str]` field
  - Add `SENTIMENTIZER_CORS_ORIGINS` env var (comma-separated)
- `sentimentizer/serve_config.yaml`:
  - Add `cors_origins: ["*"]` default

**[GAP] `ServeConfig` is a frozen dataclass.** Adding `cors_origins: list[str]` with a mutable default (`["*"]`) won't work with `frozen=True` — the `field(default_factory=lambda: ["*"])` pattern is needed. Also, the `serve_config.py` `_FIELD_TYPES` dict only supports scalar types. Parsing a comma-separated env var into a `list[str]` needs a custom coercion function. **Recommendation**: Add a `_parse_list` coercion for `list[str]` fields in `serve_config.py`.

**[REVIEW NOTE]** `cors_origins` needs `field(default_factory=lambda: ["*"])` because `ServeConfig` is frozen. For env var parsing, `SENTIMENTIZER_CORS_ORIGINS="http://localhost:3000,https://app.example.com"` splits on commas. Add `_parse_list` helper to `serve_config.py` and add to `_FIELD_TYPES` as `cors_origins: list`.

**[RISK] CORS with `@serve.ingress`.** Ray Serve processes the HTTP request before it reaches FastAPI middleware. Some Ray Serve versions have their own CORS handling. If Ray's HTTP proxy adds CORS headers, and we also add FastAPI CORS middleware, we could get duplicate headers. **Mitigation**: Test with a running Ray Serve instance after adding the middleware — specifically test `OPTIONS` preflight requests, which Ray's proxy may handle differently than FastAPI middleware expects.

**[SAFE]** The `cors_origins: ["*"]` default is appropriate for dev but must be locked down for production. The env var override (`SENTIMENTIZER_CORS_ORIGINS`) handles this cleanly.

---

### 7. Centralize Pydantic validation in model classes

**Problem**: `max_text_length` validation is duplicated manually in 5 handlers. Should be on the `Field` definition.

**Changes**:

- `sentimentizer/serve/app.py`:
  - Add `max_length` to `PredictRequest.text`, `TokenizeRequest.text`
  - Add custom validator on `BatchRequest.texts` for per-item max length and list max size
  - Remove all manual `if len(body.text) > self.cfg.max_text_length: raise HTTPException(400)` from handlers

**[RISK] Config-driven limit vs compile-time Pydantic model.** Pydantic model classes are defined at module level. `max_text_length` comes from `load_serve_config()` (also called at module level as `cfg = load_serve_config()` in `serve.py`). So we **can** use `max_length=cfg.max_text_length` on the `Field`. But there's a subtlety: the deployment's `__init__` also calls `load_serve_config()` (in `SentimentizerDeployment.__init__`), which could return different values if env vars changed between module load and deployment init. In practice this won't happen (same env in same process), but the two configs are independent objects. **Recommendation**: Use the module-level `cfg` for Pydantic models, and keep `self.cfg` in the deployment for runtime checks. Document that the Pydantic validation limit is fixed at module-load time.

**[RISK] Error message quality.** Currently, manual validation returns `Text too long (12000 chars, max 10000)` with a 400. Pydantic's built-in `max_length` returns `String should have at most 10000 characters` with a 422. The 422 status code is technically more correct (it's a validation error, not a business-logic error), but it changes the status code clients see from 400 → 422. **Mitigation**: This is the correct behavior per HTTP semantics — 422 is for validation errors. Accept the status code change.

**[GAP] Batch per-item validation.** `BatchRequest.texts[i]` length validation needs care with Pydantic `Field` semantics. `max_length` on a `list[str]` field limits the **list length** (number of items), not the **string length** of each item. For per-item string length validation, use `Annotated` types:

```python
from typing import Annotated

class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]

class BatchRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]] = Field(
        ..., min_length=1, max_length=cfg.max_batch_size
    )
```

This gives us both per-item string length validation *and* list size validation from Pydantic directly. Pydantic's error will include the index via the `loc` field (e.g., `loc: ["body", "texts", 2]`), which is arguably better than our custom `texts[2] too long` message. **Recommendation**: Use `Annotated` types as shown above. This removes **all** manual validation from handlers — both text length and batch size checks.

**[REVIEW NOTE]** 400→422 status code change is correct per HTTP semantics. Document this change for internal clients. The module-level `cfg` is used for Pydantic model definitions (fixed at import time), while `self.cfg` in the deployment is used for runtime checks. Add a comment in code noting this distinction.

---

## P2 — Nice to have

### 8. Merge `/predict` and `/batch` into one endpoint

**[RISK] `@serve.batch` interaction.** The current design uses `@serve.batch` on `predict_sentiment()` to auto-batch multiple individual `/predict` calls. If we merge `/predict` and `/batch` into one endpoint, the batch path (where the client sends N texts) already has all N texts — auto-batching within Ray is unnecessary for explicit batches. But single-text calls still benefit from auto-batching. We'd need the handler to branch: if `texts` is present, call `predict_batch` directly; if `text` is present, call `predict_sentiment` (auto-batched). This is doable but changes the request model from `PredictRequest | BatchRequest` to a union type.

**[RISK] OpenAPI schema generation.** A single endpoint that accepts either `{"text": ...}` or `{"texts": [...]}` means the request body schema has two optional fields where exactly one must be present. FastAPI can handle this with `Union[PredictRequest, BatchRequest]` and discriminated unions, but the generated OpenAPI schema will flag both fields as optional, which may confuse clients.

**Recommendation**: Defer until there's clear evidence the two-endpoint design is confusing consumers. The current design works and is well-documented.

### 9. Standardize error response envelope ✅ IMPLEMENTED

**Problem**: Error responses used four different shapes for the `detail` field depending on the error source:
  - Manual validation: `{"detail": "Text too long (12000 chars, max 10000)"}` with 400
  - Pydantic validation: `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` with 422
  - Unhandled errors: `{"detail": "Internal server error"}` with 500
  - Model not loaded: `{"detail": "Sentiment model not loaded: ..."}` with 503

**Implementation**: Defined a standard error envelope `{"error": {"code": "...", "message": "..."}}` via `http_exception_handler`. Unhandled exceptions return `{"error": {"code": "internal_error", "message": "Internal server error", "request_id": "..."}}`. HTTP exceptions with string detail are wrapped in the envelope. Pydantic 422 validation errors retain their default format. `_status_code_to_error_code()` maps common status codes to machine-readable strings (e.g., 400→"bad_request", 503→"service_unavailable").

### 10. Add `model` field to request body ✅ IMPLEMENTED

**Implementation**: Option C was chosen — `PredictRequest` and `BatchRequest` accept an optional `model: str | None = None` field. If provided, it's validated against `self.predictor.model_name` via `_validate_model()` which returns 400 on mismatch. If omitted, the default model is used. This provides the API shape for future multi-model support without the implementation cost.

### 11. Add `top_k` / `include_scores` parameters ✅ IMPLEMENTED

**Implementation**: `PredictRequest` and `BatchRequest` accept `include_scores: bool = True` and `top_k: int | None = None`. `include_scores=False` omits the `scores` key from response predictions. `top_k=N` returns only the top N classes in `scores`, sorted by probability. `_format_prediction()` static method handles the formatting logic. The `model` field on `BaseSentimentModel` already returns all 3 class probabilities `(B, 3)` tensor, so `top_k` is just a sort+slice on the output.

### 12. Add rate-limit headers

**[GAP] Rate limiting already exists at the infrastructure layer.** `k8s/ingress.yaml` line 11 already has `nginx.ingress.kubernetes.io/limit-rps: "10"` with burst multiplier 3. Ray Serve's `max_ongoing_requests=20` also limits concurrency. Adding application-level rate limiting on top of these creates three layers of rate limiting with different semantics (RPS vs concurrency vs token bucket). **Recommendation**: Before implementing app-level rate limiting, verify that the existing ingress-level limiting is insufficient. If it is, replace the ingress limit with app-level (more precise per-client tracking), don't layer them.

---

## P3 — Future

### 13. Add `GET /v1/models/{model_name}` ✅ IMPLEMENTED

**Implementation**: Added `GET /v1/models/{model_name}` endpoint returning metadata for a single model. Returns 400 for unknown model names (not in `_MODEL_CONFIGS`), 404 if model exists but isn't loaded. Uses `Path(...)` parameter for model_name.

### 14. Add token `usage` to response ✅ IMPLEMENTED

**Implementation**: `predict_batch()` now includes `token_count` in each prediction result. Token counts are computed via `len(regex_tokenize(text))` alongside the existing tokenization — no extra tokenization cost. The `_format_prediction()` handler in `serve.py` passes `token_count` through to the API response. This adds a `token_count` field to each prediction dict alongside `label`, `score`, `scores`, and `model`.

### 15. Request body size limit middleware ✅ IMPLEMENTED

**Implementation**: Added `_RequestBodySizeLimitMiddleware` (extends `BaseHTTPMiddleware`) that checks `Content-Length` header on POST/PUT/PATCH requests. Rejects bodies > 1 MiB with 413 `{"error": {"code": "request_too_large", "message": "Request body exceeds 1048576 bytes"}}`. Defense-in-depth alongside K8s ingress `proxy-body-size: "1m"`. Registered as innermost middleware (first added in code, processed last in request chain).

### 16. API key authentication

**[RISK] Auth middleware and `@serve.ingress`.** Ray Serve's HTTP proxy processes requests before they reach FastAPI. If we add auth middleware in FastAPI but Ray Serve doesn't enforce it at the proxy level, a determined attacker can bypass it by hitting the Ray Serve proxy directly. **Recommendation**: If auth is needed, implement at the ingress controller level (NGINX `nginx.ingress.kubernetes.io/auth-url`) or via a sidecar auth proxy, not in the application layer.