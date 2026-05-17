# sentimentizer

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight PyTorch models for sentiment analysis. Small models can be pretty effective for classification tasks at a much smaller cost to deploy — all models were trained on a single 2080Ti GPU in minutes, and inference requires less than 1GB of memory.

> **Beta release** — API is subject to change.

## Install
```bash
# Install local-only version (no Ray dependency)
uv add sentimentizer

# Install with distributed training, tuning, and serving features
uv add "sentimentizer[ray]"
```

## Quick Start

```python
from sentimentizer.tokenizer import get_trained_tokenizer
from sentimentizer.models.rnn import get_trained_model

model = get_trained_model(device="cpu")
tokenizer = get_trained_tokenizer()

review_text = "greatest pie ever, best in town!"
model.predict_text(review_text, tokenizer)
# >> {'negative': 0.03, 'neutral': 0.05, 'positive': 0.92}
```

`predict_text()` on `BaseSentimentModel` returns all 3 class probabilities. `predict_text()` on `BaseSentimentModel` returns all 3 class probabilities. For the serving API, responses use the additive v1 format with explicit `label`, `score`, `scores`, and `token_count` fields:

```python
from sentimentizer.predictor import SentimentPredictor

predictor = SentimentPredictor(model_name="encoder")
predictor.predict("amazing restaurant!")
# >> {"positive": 0.92, "label": "positive", "score": 0.92,
#     "scores": {"negative": 0.03, "neutral": 0.05, "positive": 0.92},
#     "token_count": 2, "model": "encoder"}

predictor.predict_batch(["Great food!", "Terrible service."])
# >> [{"positive": 0.88, "label": "positive", "score": 0.88,
#      "scores": {...}, "token_count": 2, "model": "encoder"}, ...]

For advanced use, you can also call tokenization and prediction separately:

```python
# Two-step: tokenize first, then predict
positive_ids = tokenizer.tokenize_text(review_text)
model.predict(positive_ids)
# >> tensor([[0.03, 0.05, 0.92]])  # (1, 3) probability matrix

# Tokenize without inference (for inspection)
from sentimentizer.tokenizer import regex_tokenize, text_sequencer
tokens = regex_tokenize(review_text)
token_ids = text_sequencer(tokenizer.dictionary, tokens, tokenizer.cfg.max_len)
```

Models output **3-class probabilities** (negative, neutral, positive) that sum to 1.0 per sample.

## Models

Three architectures are available:

| Model | Module | Description |
|-------|--------|-------------|
| **Encoder** ⭐ | `sentimentizer.models.encoder` | Transformer encoder with CLS token + positional encoding (4 layers, d_model=256) — **recommended** |
| **RNN** | `sentimentizer.models.rnn` | Bidirectional 2-layer LSTM (hidden=256) with GloVe embeddings — solid baseline |
| **Decoder** | `sentimentizer.models.decoder` | Encoder-Decoder Transformer with learnable query token + cross-attention (2 encoder + 4 decoder layers) |

All models output **3-class logits** `(B, 3)` with classes: negative (0), neutral (1), positive (2).

Each module exposes `get_trained_model(device, model_config=...)` to load pre-trained weights.

## Serving

### Ray Serve (Python)

> **Note:** Serving requires the `ray` extra: `uv add "sentimentizer[ray]"`

The serve command starts a Ray Serve application with FastAPI routing (`/docs` and `/redoc` available for free). It loads the **Encoder** sentiment model and the **SetFit router** at startup. Both services share the same port with route-based dispatch.

```bash
# Start with defaults (encoder model, port 8000)
make serve

# Or via CLI with options
sentimentizer serve --host 0.0.0.0 --port 8000
```

By default, the server binds to `0.0.0.0:8000`.

#### Sentiment analysis endpoints

```bash
# Single prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "the food was terrific"}'

# Batch prediction
curl -X POST http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["great pizza!", "terrible service"]}'

# Tokenize text without inference
curl -X POST http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "the food was terrific"}'

# List all sentiment models
curl http://localhost:8000/v1/models

# Single model metadata
curl http://localhost:8000/v1/models/encoder
```

Sentiment response (additive v1 format — includes explicit fields plus backward-compat dynamic key):

```json
{
  "text": "the food was terrific",
  "prediction": {
    "label": "positive",
    "score": 0.92,
    "scores": {"negative": 0.03, "neutral": 0.05, "positive": 0.92},
    "token_count": 4,
    "model": "encoder",
    "positive": 0.92
  },
  "latency_s": 0.0043
}
```

Batch response:

```json
{
  "results": [
    {
      "text": "great pizza!",
      "prediction": {
        "label": "positive", "score": 0.89,
        "scores": {"negative": 0.02, "neutral": 0.09, "positive": 0.89},
        "token_count": 2, "model": "encoder", "positive": 0.89
      }
    },
    {
      "text": "terrible service",
      "prediction": {
        "label": "negative", "score": 0.94,
        "scores": {"negative": 0.94, "neutral": 0.04, "positive": 0.02},
        "token_count": 2, "model": "encoder", "negative": 0.94
      }
    }
  ],
  "count": 2,
  "latency_s": 0.0031
}
```

#### Router (review categorization) endpoints

```bash
# Classify a single review
curl -X POST http://localhost:8000/v1/router/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "They were so careful with my celiac needs"}'

# Classify multiple reviews
curl -X POST http://localhost:8000/v1/router/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great gluten-free options!", "The waiter was rude", "Decent pizza"]}'

# Router model metadata
curl http://localhost:8000/v1/router/models
```

Router response:

```json
{
  "text": "They were so careful with my celiac needs",
  "prediction": {"category": "dietary"},
  "latency_s": 0.0031
}
```

#### Shared endpoints

```bash
# Liveness probe (always returns 200)
curl http://localhost:8000/health/live

# Readiness probe (503 if model not loaded)
curl http://localhost:8000/health/ready

# Backward-compatible health check (delegates to readiness)
curl http://localhost:8000/health

# Interactive API docs (Swagger UI)
open http://localhost:8000/docs
```

### Go CLI Client

A Go CLI client is included for interacting with the serve endpoint:

```bash
# Build and run
go run main.go -text "the food was terrific"

# Pipe input
echo "terrible service" | go run main.go

# Positional arguments
go run main.go "best restaurant in town"

# Raw JSON output
go run main.go -raw -text "amazing pasta"

# Custom endpoint
go run main.go -host http://remote:8000 -text "great coffee"
```

The client outputs colorized results with emoji indicators:

```
Text:       the food was terrific
Prediction: positive 👍
Scores:     negative=0.03, neutral=0.05, positive=0.92
Latency:    12ms
```

## Training

### Prerequisites

To retrain the model:

1. Get the Yelp [dataset](https://www.yelp.com/dataset) — download `yelp_dataset.tar` and place it in `../data/` (one level above the project root)
2. Get the GloVe 6B 100D [embeddings](https://nlp.stanford.edu/projects/glove/) — download `glove.6B.zip` and place it in `../data/` (one level above the project root)

The expected directory structure:

```
data/                            # one level above project root
├── yelp_dataset.tar             # Yelp dataset (downloaded)
└── glove.6B.zip                 # GloVe embeddings (downloaded)

torch-sentiment/                 # project root
├── sentimentizer/
│   └── data/
│       ├── yelp.dictionary      # Generated during training
│       ├── weights.pth          # Generated during training
│       └── ...
└── ...
```

### Single-node training (recommended for laptops and single-GPU machines)

```bash
# Auto-detect best device (cuda > mps > cpu)
python workflows/driver.py --device auto --type new --save

# NVIDIA GPU
python workflows/driver.py --device cuda --type new --save

# Apple Silicon (M1/M2/M3/M4) — uses Metal Performance Shaders
python workflows/driver.py --device mps --type new --save

# CPU only (slowest)
python workflows/driver.py --device cpu --type new --save

# Quick iteration with less data
python workflows/driver.py --device mps --type new --save --stop 5000
```

> **Tip:** On a single machine, single-node training is always faster than distributed. Use `--distributed` only when you have multiple GPUs.

### Distributed training with Ray Train (multi-GPU or multi-machine only)

> **Note:** Distributed training requires the `ray` extra: `uv add "sentimentizer[ray]"`

```bash
# Run with 2 workers (default)
python workflows/driver.py --device cuda --distributed --save

# Run with 4 workers
python workflows/driver.py --device cuda --distributed --num-workers 4 --save

# Run on CPU only
python workflows/driver.py --device cpu --distributed --num-workers 2
```

The `--distributed` flag enables Ray Train, which distributes data and model training across multiple workers. Each worker gets a shard of the dataset and runs the training loop with PyTorch Distributed Data Parallel (DDP). Checkpoints and metrics are aggregated automatically by Ray Train.

**Distributed training adds overhead** (process group init, gradient sync, actor management) and is slower than single-node on a single GPU. Only use it when you have multiple GPUs or machines.

### CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | Device to use: `auto` (detect), `cuda`, `mps`, or `cpu` |
| `--model` | `rnn` | Model type: `rnn`, `encoder`, or `decoder` |
| `--type` | `new` | Run type: `new` (from scratch) or `update` (resume) |
| `--stop` | `10000` | Number of lines to load from the dataset |
| `--save` | off | Save model weights after training (flag, no value needed) |
| `--distributed` | off | Enable distributed training with Ray Train (flag, no value needed) |
| `--num-workers` | `2` | Ray Train workers (distributed mode only; single-node ignores this) |
| `--agent-tune` | off | Use Pydantic AI + LangGraph agent for hyperparameter tuning (GLM 5.1 via Ollama) (flag, no value needed) |
| `--agent-config` | `None` | Path to agent config YAML (default: `sentimentizer/agent/config.yaml`) |
| `--tune` | off | Use TuningRun skill to tune hyperparameters and validate model predictions (flag, no value needed) |
| `--tune-mode` | `agent` | Tuning mode: `agent` (LLM-guided loop) or `standalone` (single Ray Tune run) |
| `--tune-samples` | `20` | Number of Ray Tune trials per tuning iteration |
| `--tune-max-iterations`| `5` | Maximum agent tuning iterations |
| `--no-validate` | off | Skip model prediction validation after tuning (flag, no value needed) |
| `--validation-threshold`| `0.75` | Minimum fraction of correct predictions to pass validation |
| `--max-retries` | `2` | Maximum re-tuning attempts if validation fails |
| `--checkpoint-dir` | `""` | Directory to save training checkpoints (empty = no checkpointing) |
| `--checkpoint-every` | `1` | Save checkpoint every N epochs (0 = disabled) |
| `--resume` | off | Resume training from the latest checkpoint in `--checkpoint-dir` (flag, no value needed) |
| `--push-to-hub` | off | Push model weights, dictionary, and model card to Hugging Face Hub after training (flag) |
| `--pull-from-hub` | off | Pull model weights from Hugging Face Hub before running (flag) |
| `--hf-repo` | `ryeyoo/sentimentizer` | Override Hugging Face repository ID |
| `--balance-classes` | off | Enable class balancing via undersampling (flag) |
| `--balance-seed` | `42` | Random seed for class balancing |
| `--weight-smoothing` | `0.5` | Class weight smoothing exponent (0=uniform, 1=full inverse frequency) |
| `--loss-type` | `cross_entropy` | Loss function: `cross_entropy` or `focal` |
| `--label-smoothing` | `0.1` | Label smoothing for CrossEntropyLoss |
| `--neutral-oversample-ratio` | `0.0` | Target neutral class ratio via oversampling (0=disabled, 0.20=20%) |

## Checkpointing

Model checkpoints save the full training state (model weights, optimizer state, scheduler state, epoch number) so you can resume training after interruptions.

### Enable checkpointing

```bash
# Save checkpoints every epoch to a directory
python workflows/driver.py --device mps --type new --checkpoint-dir checkpoints/

# Save checkpoints every N epochs (e.g., every 2 epochs)
python workflows/driver.py --device cuda --type new --checkpoint-dir checkpoints/ --checkpoint-every 2
```

This creates two types of checkpoints in `--checkpoint-dir`:
- **Periodic checkpoints**: `checkpoint_epoch_1.pth`, `checkpoint_epoch_2.pth`, etc.
- **Best model checkpoint**: `best_model.pth` (lowest validation loss seen so far)

### Resume from a checkpoint

```bash
# Resume from the latest checkpoint
python workflows/driver.py --device mps --type new --checkpoint-dir checkpoints/ --resume
```

The `--resume` flag loads the latest periodic checkpoint and restores model weights, optimizer state, and scheduler state before continuing training.

### Programmatic API

```python
from sentimentizer.trainer import save_checkpoint, load_checkpoint, latest_checkpoint

# Save a checkpoint
save_checkpoint(model, optimizer, epoch=5, path="checkpoints/ckpt.pth", val_loss=0.32)

# Find the latest checkpoint
ckpt_path = latest_checkpoint("checkpoints/")

# Load and resume
checkpoint = load_checkpoint(ckpt_path, model, optimizer, scheduler, device="cpu")
print(f"Resuming from epoch {checkpoint['epoch']}")
```

## Hyperparameter Tuning

> **Note:** Tuning requires the `ray` extra: `uv add "sentimentizer[ray]"`

Sentimentizer offers three ways to tune hyperparameters: **Standalone**, **Iterative Agent**, and **Tuning Skill**. These range from simple one-shot sweeps to LLM-guided iterative search loops with automatic model validation.

Detailed documentation for all tuning modes, including configuration and CLI usage, can be found in [docs/tuning.md](docs/tuning.md).

| | **Standalone** | **Iterative Agent** | **Tuning Skill (Fixed Workflow)** |
|---|---|---|---|
| **What it does** | Single Ray Tune + Optuna search | LangGraph-guided iterative search loop | High-level pipeline: tune → train → validate → retry |
| **LLM involved** | ❌ No | ✅ GLM 5.1 via Ollama | ✅ (in agent mode) or ❌ (in standalone mode) |
| **Iterative** | ❌ One-shot sweep | ✅ Refines search space each iteration | ✅ Refines + validates + retries |
| **Model validation** | ❌ | ❌ | ✅ Tests predictions on known examples |
| **Auto-retry on failure** | ❌ | ❌ | ✅ Re-tunes up to `max_retries` times |
| **Saves final model** | ❌ | ❌ | ✅ Trains & saves best model weights |
| **Requires Ollama** | ❌ No | ✅ Yes | Only in agent mode |
| **CLI flag** | `--tune --tune-mode standalone` | `--agent-tune` | `--tune` (defaults to agent mode) |
| **When to use** | Quick sweep, no Ollama available | You want LLM-guided search but will handle model training yourself | You want a complete end-to-end pipeline |

## Model Synchronization (Hugging Face Hub)

Sentimentizer integrates with the Hugging Face Hub for robust weight management. Each model type has its own repository with weights, dictionary, and an auto-generated model card:

| Model | Repository | Contents |
|-------|-----------|----------|
| RNN | `ryeyoo/sentimentizer-rnn` | `rnn_weights.pth`, `yelp.dictionary`, `README.md` |
| Encoder | `ryeyoo/sentimentizer-encoder` | `encoder_weights.pth`, `yelp.dictionary`, `README.md` |
| Decoder | `ryeyoo/sentimentizer-decoder` | `decoder_weights.pth`, `yelp.dictionary`, `README.md` |
| Router | `ryeyoo/sentimentizer-router` | SetFit model artifacts |

### Automatic Weight Pulling

If local weights are missing when you start training or inference, Sentimentizer will automatically attempt to pull them from the configured Hugging Face repository based on the model type.

```bash
# Pull a specific model
make download-rnn
make download-encoder
make download-decoder

# Pull all models
make pull-hub

# Pull via CLI (auto-detects per-model repo)
python workflows/driver.py --model rnn --pull-from-hub
```

### Pushing Weights and Model Cards

After a successful training or tuning run, you can push the best weights, dictionary, and an auto-generated model card to the Hub:

```bash
# Push a specific model (weights + dictionary + model card)
make upload-rnn
make upload-encoder
make upload-decoder
make upload-router

# Push all models
make push-hub

# Push via CLI after training
python workflows/driver.py --model rnn --save --push-to-hub

# Push via CLI after tuning
python workflows/driver.py --model rnn --tune --save --push-to-hub
```

The model card (README.md) includes:
- YAML metadata (license, tags, task)
- Model architecture description
- Training data info
- **Tuning metrics** (accuracy, F1, Cohen's kappa, per-class accuracy) — when pushing after tuning
- Usage instructions with code snippets
- File listing

You can override the default repository using the `--hf-repo` flag.

## Model Configuration

All model architecture parameters are configured via dataclasses in `sentimentizer/config.py`. To change layer dimensions, update the config and retrain:

```python
from sentimentizer.config import RNNConfig, EncoderConfig, DecoderConfig

# Customize RNN — e.g., larger hidden state and 3 layers
rnn_config = RNNConfig(hidden_size=512, num_layers=3, dropout=0.3)

# Customize Encoder — e.g., wider model with 8 heads
encoder_config = EncoderConfig(d_model=512, n_heads=8, n_layers=6, ff_multiplier=4)

# Customize Decoder — e.g., deeper decoder
decoder_config = DecoderConfig(d_model=512, n_heads=8, n_encoder_layers=4, n_decoder_layers=8)
```

The config flows: **`config.py` → `DriverConfig` → `new_model(model_config=...)` / `get_trained_model(device, model_config=...)` → model `__init__` sets layer dimensions**.

| Config | Parameters | Defaults |
|--------|-----------|----------|
| `RNNConfig` | `hidden_size=256`, `num_layers=2`, `dropout=0.2` | Bidirectional LSTM |
| `EncoderConfig` | `d_model=256`, `n_heads=4`, `n_layers=4`, `dropout=0.2`, `ff_multiplier=4` | Transformer encoder + CLS token |
| `DecoderConfig` | `d_model=256`, `n_heads=4`, `n_encoder_layers=2`, `n_decoder_layers=4`, `dropout=0.2`, `ff_multiplier=4` | Encoder-decoder + query token |

## Metrics

All tuning and validation outputs include comprehensive 3-class classification metrics via [`sentimentizer/metrics.py`](sentimentizer/metrics.py):

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy (correct / total) |
| `balanced_accuracy` | Mean of per-class recalls (robust to class imbalance) |
| `negative_precision` | Precision for the negative class |
| `negative_recall` | Recall for the negative class |
| `negative_f1` | F1 score for the negative class |
| `neutral_precision` | Precision for the neutral class |
| `neutral_recall` | Recall for the neutral class (critical for minority class detection) |
| `neutral_f1` | F1 score for the neutral class |
| `positive_precision` | Precision for the positive class |
| `positive_recall` | Recall for the positive class |
| `positive_f1` | F1 score for the positive class |
| `macro_f1` | Mean of per-class F1 scores (weights classes equally) |
| `weighted_f1` | Per-class F1 weighted by class frequency |
| `cohen_kappa` | Cohen's kappa coefficient (agreement beyond chance, -1 to 1) |
| `mcc` | Matthews correlation coefficient (robust to imbalance) |
| `confusion_matrix` | 3×3 confusion matrix (negative/neutral/positive) |

These metrics are computed in three places:

- **Ray Tune trials** — reported per epoch during hyperparameter search
- **Tuning Skill validation** — computed from known sentiment examples after model training
- **Programmatic API** — available via [`compute_metrics_from_model()`](sentimentizer/metrics.py) and [`compute_metrics_from_examples()`](sentimentizer/metrics.py)

```python
from sentimentizer.metrics import compute_metrics_from_model, compute_metrics_from_examples

# From a model and dataloader
metrics = compute_metrics_from_model(model, val_loader, device="cpu")
print(f"Accuracy: {metrics.accuracy:.4f}, Macro F1: {metrics.macro_f1:.4f}, Kappa: {metrics.cohen_kappa:.4f}")

# From validation result dicts
metrics = compute_metrics_from_examples(validation_results)
print(f"Neutral recall: {metrics.neutral_recall:.4f}")
print(f"Positive F1: {metrics.positive_f1:.4f}")
```

## Architecture

The pipeline consists of three stages, all powered by Ray:

1. **Extract** — Reads raw JSON data from `.zip` or `.tar` archives using `ray.data` and tokenizes text
2. **Transform** — Converts tokens to numeric sequences using `ray.data.map_batches()` and writes processed parquet
3. **Train** — Fits the model using either single-node PyTorch or distributed Ray Train with `TorchTrainer`

Inference is served via a Ray Serve deployment with FastAPI routing (see `sentimentizer/serve.py`). Endpoints are versioned under `/v1/` (e.g., `/v1/predict`, `/v1/batch`). Health probes are unversioned (`/health/live`, `/health/ready`). The API returns the additive v1 format with explicit `label`, `score`, `scores`, and `token_count` fields. The `predict_text()` method on `BaseSentimentModel` returns all 3 class probabilities (without `token_count`) — a different API surface used for validation and export.

## Docker

Build and run the containerized service:

```bash
# Build
docker build -t sentimentizer .

# Run
docker run -p 8000:8000 -p 8265:8265 sentimentizer
```

The image uses a multi-stage build with Python 3.12-slim and CPU-only PyTorch. Port 8000 serves predictions; port 8265 exposes the Ray dashboard.

## Kubernetes

Kubernetes manifests are in the `k8s/` directory:

| File | Resource | Purpose |
|------|----------|---------|
| `deployment.yaml` | Deployment | Pod template with the sentimentizer container |
| `service.yaml` | Service | ClusterIP service for internal routing |
| `hpa.yaml` | HorizontalPodAutoscaler | Auto-scaling based on CPU/memory usage |
| `ingress.yaml` | Ingress | HTTP ingress routing |
| `pdb.yaml` | PodDisruptionBudget | Minimum available replicas during disruptions |

## Development

### With uv (recommended)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. PyTorch is configured to resolve from the CPU-only wheel index by default (no NVIDIA packages), which is what CI uses. For local GPU development, see the CUDA setup instructions below.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (CPU-only PyTorch, no Ray)
uv sync

# Install with Ray distributed features
uv sync --extra ray

# Install with dev dependencies
uv sync --extra dev --extra ray
```

#### Local CUDA / GPU development

The committed lockfile resolves CPU-only PyTorch (no NVIDIA packages). To install CUDA-enabled PyTorch locally:

```bash
uv sync --no-sources-package torch
```

This ignores the CPU-only source override in `pyproject.toml` and resolves PyTorch from PyPI (with CUDA support and NVIDIA packages). Note that this will modify `uv.lock` — do not commit those changes.

### With conda

```bash
conda create -n sentimentizer
conda install pip
pip install -e .
```

## ONNX Export

Export trained models to ONNX format for CPU-optimized inference (INT8 quantization for AVX-512):

```bash
# Export RNN with quantization (recommended)
sentimentizer export --model rnn --quantize

# Export Encoder
sentimentizer export --model encoder --quantize

# Export Decoder (no quantization)
sentimentizer export --model decoder --no-quantize

# Custom output directory
sentimentizer export --model encoder --output-dir my_onnx_models/
```

ONNX artifacts are saved to `onnx_artifacts/` (gitignored) with metadata JSON alongside each model.

> **Note:** ONNX export requires the `onnx` extra: `pip install -e ".[onnx]"`

**Tolerances**: RNN uses `1e-2` (relaxed, due to masked LSTM fallback), Encoder/Decoder use `1e-4`.

## SetFit Router

A routing classifier that categorizes Yelp reviews into three categories:

| Label | Category | Description |
|-------|----------|-------------|
| 0 | Dietary | Food allergies, celiac, FODMAP, ingredient safety |
| 1 | Service | Wait times, staff behavior, reservation issues |
| 2 | General | Ambiance, price, general food quality |

### Training

> **Note:** Router training requires the `router` extra: `pip install -e ".[router]"`

```bash
# 1. Augment seed utterances with GLM 5.1 (requires Ollama running)
make router-augment

# 2. Train the router
make router-train

# 3. Evaluate (similarity matrix + threshold calibration)
make router-evaluate

# Or run the full pipeline sequentially:
make router-pipeline
```

The `augment` command supports options for model, variations per seed, and Ollama URL via CLI:

```bash
# Customize augmentation
sentimentizer router augment --model glm-5.1:cloud --variations 30 --output my_data.jsonl

# Point to a remote Ollama instance
sentimentizer router augment --ollama-url http://remote:11434/api/generate
```

**Evaluation targets**: inter-class similarity < 0.65, intra-class similarity > 0.85.

The default base model is `BAAI/bge-base-en-v1.5` (109M params, 768-dim embeddings, strong MTEB scores). Switch to `mxbai-embed-large-v1` only if evaluation thresholds are not met:

```bash
sentimentizer router train --data augmented_yelp.jsonl --base-model mxbai-embed-large-v1
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run only Ray Train tests
uv run pytest tests/ -v -k "Ray"

# Run with coverage
uv run pytest tests/ -v --cov=sentimentizer --cov-report=term-missing
```

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
├── loader.py             # Data loading utilities
├── losses.py             # FocalCrossEntropyLoss for 3-class training
├── metrics.py            # 3-class classification metrics (per-class P/R/F1, balanced accuracy, MCC)
├── metrics_publisher.py   # Epoch metrics publishing (Prometheus + JSON)
├── predictor.py           # SentimentPredictor (model loading, inference, additive v1 format)
├── serve.py              # Ray Serve deployment: FastAPI + @serve.ingress, /v1/ prefix
├── serve_base.py          # ServiceMetrics (request/latency tracking), _DummyServe fallback
├── serve_config.py        # Serve deployment configuration (YAML/env var loading, incl. cors_origins)
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
│   └── skill.py          # TuningRun skill (tune → train → validate → retry pipeline)
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
    ├── rnn.py            # Bidirectional LSTM (3-class output)
    ├── encoder.py         # Transformer encoder model (3-class output)
    └── decoder.py         # Encoder-decoder transformer (3-class output)
```

## License

[MIT](LICENSE)