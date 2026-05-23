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

The `serve` command starts a Ray Serve application with FastAPI routing (featuring interactive Swagger docs at `/docs`). It loads the sentiment model (defaulting to the configuration in `serve_config.yaml`) and the SetFit router at startup. Both services share the same port and handle incoming requests via route-based dispatch.

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
