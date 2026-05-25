# Serving and REST API

This document describes how to deploy and interact with the unified Sentimentizer REST API.

## Ray Serve Deployment (Python)

> [!NOTE]
> Serving requires the `ray` extra. You can install it using:
> ```bash
> uv sync --extra ray
> # or if adding to a project:
> uv add "sentimentizer[ray]"
> ```

The `serve` command starts a Ray Serve application with FastAPI routing (featuring interactive Swagger docs at `/docs`). It loads the sentiment model (defaulting to the configuration in `serve_config.yaml`) and the SetFit router at startup. Image generation (Stable Diffusion 2.1, FLUX.1-dev) can be enabled via configuration. All services share the same port and handle incoming requests via route-based dispatch.

### Starting the Server

```bash
# Start with defaults (e.g., encoder model, host 0.0.0.0, port 8000)
make serve

# Or start via CLI with custom options
sentimentizer serve --host 0.0.0.0 --port 8000
```

By default, the server binds to `0.0.0.0:8000`.

---

## API Endpoints Reference

### Sentiment Analysis Endpoints

#### Single Sentiment Prediction
- **Route**: `POST /v1/predict`
- **Request Body**:
  ```json
  {
    "text": "the food was terrific"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "the food was terrific"}'
  ```
- **Response**:
  ```json
  {
    "prediction": {
      "label": "positive",
      "score": 0.92,
      "token_count": 4,
      "model": "encoder"
    },
    "latency_s": 0.0043
  }
  ```

#### Batch Sentiment Prediction
- **Route**: `POST /v1/batch`
- **Request Body**:
  ```json
  {
    "texts": ["great pizza!", "terrible service"]
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["great pizza!", "terrible service"]}'
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "prediction": {
          "label": "positive",
          "score": 0.89,
          "token_count": 2,
          "model": "encoder"
        }
      },
      {
        "prediction": {
          "label": "negative",
          "score": 0.94,
          "token_count": 2,
          "model": "encoder"
        }
      }
    ],
    "count": 2,
    "latency_s": 0.0031
  }
  ```

#### Standalone Tokenization (No Inference)
- **Route**: `POST /v1/tokenize`
- **Request Body**:
  ```json
  {
    "text": "the food was terrific"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/tokenize \
    -H "Content-Type: application/json" \
    -d '{"text": "the food was terrific"}'
  ```
- **Response**:
  ```json
  {
    "text": "the food was terrific",
    "tokens": ["the", "food", "was", "terrific"],
    "token_ids": [4, 12, 10, 48],
    "token_count": 4
  }
  ```

#### List All Sentiment Models
- **Route**: `GET /v1/models`
- **Command**:
  ```bash
  curl http://localhost:8000/v1/models
  ```

#### Single Model Metadata
- **Route**: `GET /v1/models/{model_name}`
- **Command**:
  ```bash
  curl http://localhost:8000/v1/models/encoder
  ```

---

### Router (Review Categorization) Endpoints

#### Classify Single Review
- **Route**: `POST /v1/router/predict`
- **Request Body**:
  ```json
  {
    "text": "They were so careful with my celiac needs"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/router/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "They were so careful with my celiac needs"}'
  ```
- **Response**:
  ```json
  {
    "prediction": {
      "label": "dietary",
      "score": 0.95,
      "token_count": 8
    },
    "latency_s": 0.0031
  }
  ```

#### Classify Batch of Reviews
- **Route**: `POST /v1/router/batch`
- **Request Body**:
  ```json
  {
    "texts": ["Great gluten-free options!", "The waiter was rude", "Decent pizza"]
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/router/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Great gluten-free options!", "The waiter was rude", "Decent pizza"]}'
  ```

#### Router Model Metadata
- **Route**: `GET /v1/router/models`
- **Command**:
  ```bash
  curl http://localhost:8000/v1/router/models
  ```

---

### Image Generation Endpoints

> [!NOTE]
> Image generation requires the `diffusion` extra and a GPU. You can install it using:
> ```bash
> uv sync --extra ray --extra diffusion
> ```
> Enable image generation by setting `SENTIMENTIZER_SD_ENABLED=1`, `SENTIMENTIZER_FLUX_ENABLED=1`,
> and/or `SENTIMENTIZER_SD35_ENABLED=1`, and provide API keys via `SENTIMENTIZER_API_KEYS`. Image routes require authentication;
> sentiment and router routes remain unauthenticated.

Image generation uses separate GPU-backed Ray Serve deployments (SDDeployment, FluxDeployment, SD35Deployment)
behind a lightweight CPU dispatcher (ImagesDispatcher). The [diffusion serving plan](diffusion_serving_plan.md)
has full architectural details.

#### Synchronous Image Generation
- **Route**: `POST /v1/images/generate`
- **Auth**: `Authorization: Bearer <api_key>` (required)
- **Request Body**:
  ```json
  {
    "prompt": "a red apple on a wooden table",
    "model": "sd",
    "width": 1024,
    "height": 1024,
    "output_format": "png"
  }
  ```
- **Command**:
  ```bash
  curl -X POST http://localhost:8000/v1/images/generate \
    -H "Authorization: Bearer test-key-123" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "a red apple on a wooden table", "model": "sd"}'
  ```
- **Response**:
  ```json
  {
    "id": "img_ABCDEFGHIJKL",
    "created": 1700000000,
    "model": "sd",
    "image_b64": "...",
    "format": "png",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "steps": 30,
    "guidance_scale": 7.5,
    "latency_s": 2.5
  }
  ```

Supported parameters: `prompt` (required), `model` (`"sd"`, `"flux"`, or `"sd35"`), `negative_prompt`,
`steps`, `guidance_scale`, `width`, `height`, `seed`, `response_format` (`"b64_json"` or `"url"`),
`output_format` (`"png"`, `"webp"`, `"jpeg"`), `user` (opaque abuse-tracking ID),
`Idempotency-Key` header (deduplication).

Rate-limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) are
returned on every response. Exceeding the limit returns `429` with a `Retry-After` header.

#### List Image Models
- **Route**: `GET /v1/images/models`
- **Auth**: Required
- **Command**:
  ```bash
  curl http://localhost:8000/v1/images/models \
    -H "Authorization: Bearer test-key-123"
  ```

#### Single Image Model Metadata
- **Route**: `GET /v1/images/models/{name}`
- **Auth**: Required
- **Command**:
  ```bash
  curl http://localhost:8000/v1/images/models/sd \
    -H "Authorization: Bearer test-key-123"
  ```

---

### Shared & Infrastructure Endpoints

#### Liveness Probe
- **Route**: `GET /health/live`
- **Purpose**: Returns `200 OK` (e.g. `{"status": "alive", "uptime_s": 12.3}`) to confirm that the server process is running.
- **Command**:
  ```bash
  curl http://localhost:8000/health/live
  ```

#### Readiness Probe
- **Route**: `GET /health/ready`
- **Purpose**: Returns `200 OK` if the models are successfully loaded and ready to serve traffic, otherwise returns `503 Service Unavailable`.
- **Command**:
  ```bash
  curl http://localhost:8000/health/ready
  ```

#### Backward-Compatible Health Check
- **Route**: `GET /health`
- **Purpose**: Delegates to the readiness probe.
- **Command**:
  ```bash
  curl http://localhost:8000/health
  ```

#### Interactive API Docs (Swagger UI)
- Open `http://localhost:8000/docs` in your browser to view and interact with the REST API documentation.

---

## Ray Serve Internals

This section documents non-obvious Ray Serve behaviors that affect development and operations. For the full architectural detail, see `docs/serve-api-plan.md` under "Ray Serve Architecture & Pitfalls".

### `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`

**Must be set before any Ray import.** Without it, Ray workers try to create isolated venvs via `uv` that lack the `ray` package, causing `ModuleNotFoundError: No module named 'ray'`. Set in three places:

1. `sentimentizer/serve/app.py` — module-level, before `from ray import serve`
2. `sentimentizer/serve/app.py` — `main()` function, before `ray.init()`
3. `workflows/cli.py` — `serve_cmd()`, before importing `sentimentizer.serve`
4. `sentimentizer/lifecycle.py` — module-level (covers training paths)

If you add a new entry point that uses Ray, add this env var.

### `@serve.batch` — callers get a single dict, not a list

`@serve.batch` collects multiple calls into a batch. The batch function signature is `list[dict] → list[dict]`, but **individual callers** receive only their single item from the result list. Never index into the return (e.g., `result[0]`) — it's already unwrapped.

```python
# Handler: calls batch function with single item
prediction = await self.predict_sentiment({"text": body.text})
# prediction is a dict, NOT a list[dict]

# Batch function: processes a batch
@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
async def predict_sentiment(self, inputs: list[dict]) -> list[dict]:
    # Must return same length as inputs
    ...
```

**If the return list length doesn't match inputs length, remaining callers hang forever.**

### `@serve.ingress` — no `APIRouter`, no `__call__`

- All routes must be registered on the root `FastAPI` app with explicit `/v1/...` prefixes. `APIRouter` sub-apps don't work with `@serve.ingress`.
- Defining `__call__` on the deployment class raises `RuntimeError`.

### Model loading and readiness

`SentimentizerDeployment.__init__()` loads models synchronously. If loading fails, `self._ready` stays `False` and `/health/ready` returns 503. The process stays alive — it doesn't crash. Check `/health/ready` to confirm the model loaded successfully.

### `asyncio.to_thread()` for sync inference

PyTorch model inference is blocking. All inference calls use `asyncio.to_thread()` to run in a thread pool, keeping the event loop responsive. Never call `predictor.predict_batch()` directly from an async handler.

### Deployment configuration

```python
@serve.deployment(
    num_replicas=1,          # Scale for CPU-bound load
    max_ongoing_requests=20,  # Backpressure: 21st request gets 503
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
```

For GPU inference, change `"num_gpus": 1` and set `device="cuda"`.

### `_DummyServe` fallback

`sentimentizer/serve/base.py` provides a `_DummyServe` that lets the serve module be imported without Ray installed. Always import `serve` from `base.py`:

```python
from sentimentizer.serve.base import serve  # grace fallback without ray
# NOT: from ray import serve  # ImportError without ray
```

### Startup sequence

1. `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` — prevents worker venv issues
2. `ray.init(namespace="sentimentizer", runtime_env={"py_executable": sys.executable})` — uses current Python
3. `serve.start(http_options=HTTPOptions(...))` — starts HTTP proxy
4. `serve.run(SentimentizerDeployment.bind(), ...)` — deploys the app
5. Model loading in `__init__()` — `/health/ready` returns 503 until models load

---

## Go CLI Client

A lightweight Go CLI client is included in the project root (`main.go`) to interact directly with the REST endpoints from the command line.

### Compilation and Usage

```bash
# Run with single text
go run main.go -text "the food was terrific"

# Positional arguments (defaults to single prediction)
go run main.go "best restaurant in town"

# Pipe input from stdin
echo "terrible service" | go run main.go

# Output raw JSON responses
go run main.go -raw -text "amazing pasta"

# Point to a custom serve endpoint
go run main.go -host http://remote-host:8000 -text "great coffee"
```

### Example Console Output

The Go client outputs colorized results with emojis indicating predicted sentiment classes:

```
Text:       the food was terrific
Prediction: positive 👍
Scores:     negative=0.03, neutral=0.05, positive=0.92
Latency:    12ms
```
