# sentimentizer

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight PyTorch models for sentiment analysis. Small models can be pretty effective for classification tasks at a much smaller cost to deploy — all models were trained on a single GPU in minutes, and inference requires less than 1GB of memory.

> **Beta release** — API is subject to change.

---

## Install

```bash
# Install local-only version (no Ray dependency)
uv add sentimentizer

# Install with distributed training, tuning, and serving features
uv add "sentimentizer[ray]"

# Install with image generation (Stable Diffusion / FLUX)
uv add "sentimentizer[diffusion]"

# Install both
uv add "sentimentizer[ray,diffusion]"
```

---

## Quick Start
  
Run a pre-trained model locally:

```python
from sentimentizer.predictor import SentimentPredictor

# Load the model
predictor = SentimentPredictor(model_name="encoder")

# Predict sentiment (returns label, score, token count, and model type)
result = predictor.predict("amazing restaurant!")
# >> {"label": "positive", "score": 0.92, "token_count": 2, "model": "encoder"}

# Batch prediction
results = predictor.predict_batch(["Great food!", "Terrible service."])
# >> [{"label": "positive", "score": 0.88, "token_count": 2, "model": "encoder"}, ...]
```

Models output **3-class probabilities** (negative, neutral, positive) that sum to 1.0 per sample.

---

## Image Generation (SD 2.1 / FLUX.1-dev / SD 3.5 Medium)

The diffusion serving pipeline adds GPU-backed image generation endpoints alongside sentiment analysis. Disabled by default; enable via config.

### Prerequisites

```bash
# Install with diffusion support (includes diffusers, transformers, accelerate, safetensors)
uv sync --extra diffusion

# For GPU: install CUDA-enabled PyTorch
uv sync --no-sources-package torch
```

### Configure

Set environment variables or edit `sentimentizer/serve/serve_config.yaml`:

```yaml
# Enable one or more models
sd_enabled: true
flux_enabled: false            # FLUX needs ~22 GB VRAM (L4/A100)
sd35_enabled: false            # SD 3.5 Medium needs ~5-6 GB VRAM
default_image_model: "sd"      # used when request omits model field

# Auth — required for image routes (/v1/images/*)
api_keys: ["sk-your-secret-key"]

# Optional: custom model paths
sd_model_id: "stabilityai/stable-diffusion-2-1"
sd35_model_id: "stabilityai/stable-diffusion-3.5-medium"
flux_model_path: "/path/to/flux1-dev-q8_0.gguf"   # GGUF quantized weights
```

Or via environment variables:

```bash
export SENTIMENTIZER_SD_ENABLED=true
export SENTIMENTIZER_API_KEYS=sk-your-secret-key
```

### Run

```bash
# Start the Ray Serve deployment (loads model on startup)
python -m sentimentizer.serve

# SD generation (sync, ~2-3s on L4)
curl -X POST http://localhost:8000/v1/images \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red apple on a wooden table", "model": "sd", "width": 512, "height": 512}'

# List available models
curl http://localhost:8000/v1/images/models \
  -H "Authorization: Bearer sk-your-secret-key"

# Async job mode (for FLUX or long-running requests)
curl -X POST http://localhost:8000/v1/images/jobs \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic portrait of an astronaut", "model": "flux"}'

# Poll job status
curl http://localhost:8000/v1/images/jobs/{job_id} \
  -H "Authorization: Bearer sk-your-secret-key"
```

### MPS (Apple Silicon) Support

SD 2.1 and SD 3.5 Medium work on MPS devices. FLUX GGUF is CUDA-only; setting `flux_enabled: true` on MPS raises `RuntimeError` at startup.

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/images` | Required | Synchronous image generation |
| POST | `/v1/images/jobs` | Required | Async job creation (201 + Location) |
| GET | `/v1/images/jobs` | Required | List jobs (paginated, scoped to API key) |
| GET | `/v1/images/jobs/{id}` | Required | Get job status |
| DELETE | `/v1/images/jobs/{id}` | Required | Cancel job (best-effort) |
| GET | `/v1/images/models` | Required | List available image models |
| GET | `/v1/images/models/{name}` | Required | Single model metadata |

---

## Models

Four architectures are available:

| Model | Module | Description |
| :--- | :--- | :--- |
| **ModernBERT** ⭐ | `sentimentizer.models.modernbert` | ModernBERT contextual transformer backbone with mean pooling and layer-wise unfreezing — **best performance** |
| **Encoder** | `sentimentizer.models.encoder` | Transformer encoder with CLS token + positional encoding (4 layers, `d_model=256`) |
| **RNN** | `sentimentizer.models.rnn` | Bidirectional 2-layer LSTM (`hidden=256`) with pre-trained GloVe embeddings — solid baseline |
| **Decoder** | `sentimentizer.models.decoder` | Encoder-Decoder Transformer with learnable query token + cross-attention (2 encoder + 4 decoder layers) |

All models output **3-class logits** `(B, 3)` mapped to: negative (0), neutral (1), positive (2).

---

## Documentation

Detailed guides and implementation details are available in the specialized documentation files:

- 🚀 **[Model Serving Guide](docs/serving.md)**: Ray Serve application deployment, FastAPI endpoints (sentiment/routing/image generation), and the Go CLI client.
- 🎨 **[Diffusion Serving Plan](docs/diffusion_serving_plan.md)**: Image generation API design (SD 2.1, FLUX.1-dev, SD 3.5 Medium), middleware (auth, rate limiting, idempotency), and GPU deployment.
- 🏋️ **[Model Training & Checkpointing Guide](docs/training.md)**: Yelp datasets, single-node/distributed commands, training arguments, sleep prevention, and checkpoint resuming.
- ⚙️ **[Model Configuration Reference](docs/configuration.md)**: Configuration dataclasses (`RNNConfig`, `EncoderConfig`, etc.), parameter defaults, and consistency checks.
- 🎛️ **[Hyperparameter Tuning Guide](docs/tuning.md)**: Optuna searches, LangGraph iterative agent tuning (via Ollama GLM 5.1), and validation/retries.
- 🔗 **[Hugging Face Hub Integration](docs/huggingface.md)**: Pre-trained weights synchronization, explicit pull/push, and auto-generated model cards.
- 📈 **[Metrics and Monitoring Pipeline](docs/metrics.md)**: Exporter details, Grafana dashboards, Prometheus scrape targets, NaN handling, and real-time intra-epoch batch metrics.
- 🧭 **[SetFit Review Router](docs/router.md)**: Utterance classification categories (Dietary/Service/General), Ollama GLM 5.1 augmentation, training, and evaluation.
- 🛠️ **[Troubleshooting Guide](docs/troubleshooting.md)**: Solutions for common issues like majority-class collapses, vocabulary matches, or scheduling.

---

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (CPU-only PyTorch, no Ray)
uv sync

# Install with Ray distributed features
uv sync --extra ray

# Install dev and test suites
uv sync --extra dev --extra ray

# Install with diffusion (image generation) support
uv sync --extra diffusion

# Full development install
uv sync --extra dev --extra ray --extra diffusion
```

### Local CUDA / GPU development
The locked packages resolve CPU-only PyTorch. To install CUDA-enabled PyTorch locally:
```bash
uv sync --no-sources-package torch
```
*Note: This ignores CPU overrides in `pyproject.toml` and pulls PyTorch from PyPI with CUDA/NVIDIA libraries. Avoid committing changes to `uv.lock`.*

---

## Testing

Ensure local CI tests pass prior to submitting changes:

```bash
# Run all tests
uv run pytest tests/ -v

# Run only Ray Train tests
uv run pytest tests/ -v -k "Ray"

# Run with coverage report
uv run pytest tests/ -v --cov=sentimentizer --cov-report=term-missing
```

---

## Project Structure

```
sentimentizer/
├── __init__.py          # Logging and timing utilities
├── compat.py            # Transformers/setfit compatibility shims
├── config.py            # Configuration dataclasses and constants
├── data_source.py       # Unified DataSource protocol (pandas/Ray)
├── device.py            # Device detection (cuda/mps/cpu)
├── env.py               # Environment setup (NVIDIA LD_LIBRARY_PATH)
├── extractor.py          # Ray Data extraction from zip/tar archives
├── exporter.py           # Standalone Prometheus metrics exporter
├── export_onnx.py        # ONNX export, quantization, validation
├── hf.py                # Hugging Face Hub push/pull + model card generation
├── hf_dataset.py        # Dataset wrapper and collation for HF transformers
├── hf_tokenizer.py      # Tokenizer wrapper for HF transformers
├── loader.py             # Data loading utilities
├── losses.py             # FocalCrossEntropyLoss for 3-class training
├── metrics.py            # 3-class classification metrics (per-class P/R/F1, balanced accuracy, MCC)
├── metrics_publisher.py   # Epoch metrics publishing (Prometheus + JSON) + intra-epoch batch snapshots
├── predictor.py           # SentimentPredictor (model loading, inference)
├── safety.py              # Shared prompt safety (NSFW blocklist, injection patterns)
├── serve/                 # Ray Serve deployment: FastAPI + @serve.ingress, /v1/ prefix
│   ├── app.py             # FastAPI route handlers and deployment class
│   ├── base.py            # ServiceMetrics (request/latency tracking), _DummyServe fallback
│   ├── config.py           # Serve deployment configuration (YAML/env var loading, incl. cors_origins)
│   ├── middleware.py       # Auth, rate limiting, idempotency, prompt safety for image routes
│   ├── models.py          # Pydantic request/response models for Swagger docs
│   ├── diffusion_models.py # Pydantic request/response models for image generation (+ Job models)
│   └── diffusion_app.py    # SD/FLUX/SD35 deployments + ImagesDispatcher routes + job endpoints
├── diffusion/              # Diffusion model loading + inference
│   ├── config.py           # DiffusionModelConfig, SD/FLUX/SD35 default configs
│   ├── job_store.py         # JobStoreLogic + Ray actor for async job metadata
│   └── predictor.py         # DiffusionPredictor ABC, SDPredictor, FluxPredictor, SD35Predictor
├── tokenizer.py           # Text tokenizer with pre-trained support
├── trainer.py             # Training logic
├── tuner.py               # Ray Tune + Optuna hyperparameter search
├── data/                  # Training data (Yelp, GloVe)
├── agent/                 # LLM-guided tuning agent
│   ├── __init__.py       # Package exports
│   ├── config.yaml       # Agent + tuner configuration (YAML)
│   ├── loader.py         # YAML → dataclass config loader
│   ├── models.py         # Pydantic models (AnalysisResult, TuningDecision, etc.)
│   ├── agents.py         # Pydantic AI agents (GLM 5.1 via Ollama)
│   ├── prompts.py        # System prompts for analysis & strategy agents
│   ├── state.py          # LangGraph AgentState TypedDict
│   ├── nodes.py          # LangGraph node functions (analyze, decide, tune, evaluate)
│   ├── graph.py          # LangGraph StateGraph + run_agent_tuning() entry point
│   └── diagnose_model.py # TuningRun workflow (tune → train → validate → retry pipeline)
├── router/                # SetFit router module
│   ├── __init__.py       # Package exports
│   ├── config.py         # SetFitConfig, RouteLabels, AugmentConfig
│   ├── seeds.py          # Golden example utterances per category
│   ├── augment.py        # GLM 5.1 augmentation via Ollama
│   ├── dataset.py        # JSONL dataset loader, train/test split
│   ├── train_router.py   # SetFit training with compat shims
│   └── evaluate.py       # Similarity heatmap, threshold calibration
└── models/
    ├── __init__.py
    ├── base.py            # BaseSentimentModel with predict() and predict_text()
    ├── hf_base.py         # Base class for Hugging Face transformer architectures
    ├── rnn.py            # Bidirectional LSTM (3-class output)
    ├── encoder.py         # Transformer encoder model (3-class output)
    ├── decoder.py         # Encoder-decoder transformer (3-class output)
    └── modernbert.py      # ModernBERT transformer classifier wrapper (3-class output)
```

---

## License

[MIT](LICENSE)
