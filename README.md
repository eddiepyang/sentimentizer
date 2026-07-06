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

# Install with hardware-accelerated image generation on Apple Silicon (via mflux)
uv add "sentimentizer[mlx-diffusion]"

# Install all features
uv add "sentimentizer[ray,diffusion,mlx-diffusion]"
```

---

## Quick Start
  
Run a pre-trained model locally:

```python
from sentimentizer.predictor import SentimentPredictor

# Load the model
predictor = SentimentPredictor(model_name="encoder")

# Predict sentiment (returns label, score, all-class scores, token count, and model type)
result = predictor.predict("amazing restaurant!")
# >> {"positive": 0.848753035068512,
#     "label": "positive",
#     "score": 0.848753035068512,
#     "scores": {"negative": 0.05746544152498245,
#                "neutral": 0.09378153085708618,
#                "positive": 0.848753035068512},
#     "token_count": 2,
#     "model": "encoder"}

# Batch prediction
results = predictor.predict_batch(["Great food!", "Terrible service."])
# >> [{'positive': 0.848753035068512,
#      'label': 'positive',
#      'score': 0.848753035068512,
#      'scores': {'negative': 0.05746544152498245,
#                 'neutral': 0.09378153085708618,
#                 'positive': 0.848753035068512},
#      'token_count': 2,
#      'model': 'encoder'},
#     {'negative': 0.9586858749389648,
#      'label': 'negative',
#      'score': 0.9586858749389648,
#      'scores': {'negative': 0.9586858749389648,
#                 'neutral': 0.03224344924092293,
#                 'positive': 0.0090706842020154},
#      'token_count': 2,
#      'model': 'encoder'}]
```

Models output **3-class probabilities** (negative, neutral, positive) that sum to 1.0 per sample.

---

## Image Generation (SD 3.5 Medium / FLUX.2 Klein / SDXL)

The diffusion serving pipeline adds GPU-backed image generation endpoints alongside sentiment analysis. Disabled by default; enable via config. Three models are supported: **SD 3.5 Medium** (Stability Community License, 1024² flagship), **FLUX.2 Klein 4B** (Apache 2.0, step-distilled, ~13 GB VRAM), and **SDXL** with multi-slot support for drop-in fine-tunes (Juggernaut XL, Illustrious XL, etc.). On Apple Silicon Macs, the FLUX.2 Klein model can be hardware-accelerated using the native MLX framework (via `mflux`), which yields a **4-5x speedup** (~3.7s per image).

### Prerequisites

```bash
# Install with standard PyTorch diffusers support:
uv sync --extra diffusion

# For MLX acceleration on Apple Silicon (FLUX.2 Klein only):
uv sync --extra diffusion --extra mlx-diffusion

# For CUDA GPU: install CUDA-enabled PyTorch
uv sync --no-sources-package torch
```

### Configure

Configuration lives in two YAML files following the same pattern (dataclass defaults < YAML values < environment variable overrides):

- **`sentimentizer/serve/serve_config.yaml`** — operational settings: which models to enable, API keys, rate limits, model IDs, CPU offload mode.
- **`sentimentizer/diffusion/diffusion_config.yaml`** — model-internal defaults: denoising steps, guidance scale, max pixels, dimension alignment.

Edit `serve_config.yaml` to enable a model:

```yaml
# Enable one or more models
sd35_enabled: true               # SD 3.5 Medium (~10 GB VRAM)
flux2_klein_enabled: false       # FLUX.2 Klein 4B (~13 GB VRAM, Apache 2.0)
sdxl_models: []                  # Named SDXL slots: ["anime:John6666/noob-sdxl-v10", ...]
default_image_model: "sd35"      # used when request omits model field

# Auth — required for image routes (/v1/images/*)
api_keys: ["sk-your-secret-key"]

# Optional: override model IDs
sd35_model_id: "stabilityai/stable-diffusion-3.5-medium"
flux2_klein_model_id: "black-forest-labs/FLUX.2-klein-4B"

# Optional: CPU offload for VRAM-constrained GPUs
# "" (default, full GPU), "model" (whole-module swap), "sequential" (submodule swap)
sd35_cpu_offload: ""
flux2_klein_cpu_offload: ""

# Optional: backend to use for FLUX.2 Klein
# "auto" (default, MLX on Apple Silicon, diffusers otherwise), "diffusers", or "mlx"
flux2_klein_backend: "auto"
```

Or via environment variables:

```bash
# Enable SD 3.5 Medium
export SENTIMENTIZER_SD35_ENABLED=true
export SENTIMENTIZER_API_KEYS=sk-your-secret-key

# Enable FLUX.2 Klein
export SENTIMENTIZER_FLUX2_KLEIN_ENABLED=true

# Enable one or more SDXL slots (comma-separated name:model_id list)
export SENTIMENTIZER_SDXL_MODELS="anime:John6666/noob-sdxl-v10,base:stabilityai/stable-diffusion-xl-base-1.0"

# Optional: cap VRAM with CPU offload (see "Low VRAM" below)
export SENTIMENTIZER_SD35_CPU_OFFLOAD=sequential

# Optional: backend to use for FLUX.2 Klein ("auto", "diffusers", or "mlx")
export SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND=auto
```

### Run

```bash
# Start the Ray Serve deployment (loads model on startup)
python -m sentimentizer.serve

# SD 3.5 Medium generation (sync, ~4-6s on L4)
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic portrait of an astronaut", "model": "sd35", "width": 1024, "height": 1024}'

# FLUX.2 Klein generation (sync, ~1-3s on a fitting GPU, 4 steps)
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a calico cat in a teacup, soft window light", "model": "flux2_klein", "width": 1024, "height": 1024}'

# SDXL slot generation (model name matches an entry from sdxl_models)
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a watercolor still life", "model": "anime", "width": 1024, "height": 1024}'

# List available models
curl http://localhost:8000/v1/images/models \
  -H "Authorization: Bearer sk-your-secret-key"

# Async job mode (for long-running requests)
curl -X POST http://localhost:8000/v1/images/jobs \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cinematic portrait of an astronaut", "model": "sd35"}'

# Poll job status
curl http://localhost:8000/v1/images/jobs/{job_id} \
  -H "Authorization: Bearer sk-your-secret-key"
```

### Low VRAM (SD 3.5 CPU offload)

SD 3.5 Medium peak VRAM is ~10 GB at 1024×1024 fp16, which won't fit comfortably on 8–11 GB GPUs (e.g. 2080 Ti, 3060). Enable diffusers' CPU offload via `SENTIMENTIZER_SD35_CPU_OFFLOAD` (or `sd35_cpu_offload` in `serve_config.yaml`):

| Mode | Peak VRAM | Latency vs. baseline | When to use |
|------|-----------|----------------------|-------------|
| `""` (default) | ~10 GB | 1.0× | Plenty of VRAM (12 GB+) |
| `model` | ~5–9 GB | ~1.1–1.3× | Tight but workable (10–12 GB) |
| `sequential` | ~1–2 GB | ~3–5× | Very tight (8 GB or less, shared GPU) |

The selected mode is logged at warmup as `cpu_offload=<mode>` so you can confirm it took effect.

### MPS (Apple Silicon) Support

- **SD 3.5 Medium and SDXL**: Run on MPS devices in fp16 using `diffusers`.
- **FLUX.2 Klein**: Can run on MPS via `diffusers` (slow: ~18-20s per image) or via **MLX** (fast: **~3.7s steady-state** with a ~5s cold start on M3 Ultra). Install with `uv sync --extra mlx-diffusion` and set `flux2_klein_backend: "auto"` or `"mlx"`. CPU offloading and dtype parameters are ignored by the MLX backend because MLX manages precision and uses unified system memory.

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
- 🎨 **[Diffusion Serving Plan](docs/diffusion_serving_plan.md)**: Image generation API design (SD 3.5 Medium, FLUX.2 Klein, SDXL slots), middleware (auth, rate limiting, idempotency), and GPU deployment.
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

# Full development install with MLX support
uv sync --extra dev --extra ray --extra diffusion --extra mlx-diffusion
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
├── diffusion/                # Diffusion model loading + inference
│   ├── config.py             # DiffusionModelConfig + load_diffusion_config() (YAML + env-var overrides)
│   ├── diffusion_config.yaml # SD35 / SDXL / FLUX.2 Klein defaults (steps, guidance, cpu_offload)
│   ├── job_store.py          # JobStoreLogic + Ray actor for async job metadata
│   ├── mlx_compat.py         # MFLUX_AVAILABLE and is_mlx_device() guards
│   ├── mlx_predictor.py      # MLXFlux2KleinPredictor implementation (no torch dependency)
│   └── predictor.py          # DiffusionPredictor ABC, SD35Predictor, SDXLPredictor, Flux2KleinPredictor, create_predictor()
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
