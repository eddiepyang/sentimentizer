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
positive_ids = tokenizer.tokenize_text(review_text)
model.predict(positive_ids)
# >> tensor(0.9701)
```

Scores range from **0** (very negative) to **1** (very positive).

## Models

Three architectures are available:

| Model | Module | Description |
|-------|--------|-------------|
| **Encoder** ⭐ | `sentimentizer.models.encoder` | Transformer encoder with CLS token + positional encoding (4 layers, d_model=256) — **recommended** |
| **RNN** | `sentimentizer.models.rnn` | Bidirectional 2-layer LSTM (hidden=256) with GloVe embeddings — solid baseline |
| **Decoder** | `sentimentizer.models.decoder` | Encoder-Decoder Transformer with learnable query token + cross-attention (2 encoder + 4 decoder layers) |

**Why Encoder?** Self-attention over the full token sequence with a CLS token is the most natural fit for sentence-level classification. The RNN processes tokens sequentially and can miss long-range dependencies, though bidirectionality helps. The Decoder uses cross-attention (a query token attends to encoded text), which is effective but adds encoder overhead — best reserved for cases where you want the Decoder's cross-attention pattern.

Each module exposes `get_trained_model(device, model_config=...)` to load pre-trained weights.

## Serving

### Ray Serve (Python)

> **Note:** Serving requires the `ray` extra: `uv add "sentimentizer[ray]"`

The `sentimentizer/serve.py` entry point deploys a Ray Serve application that loads **all three models** (RNN, Encoder, Decoder) at startup. You can select which model to use per request via the `model` field.

```bash
uv run serve run sentimentizer.serve:app --host 0.0.0.0 --port 8000
```

Send a prediction request (defaults to RNN):

```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"text": "the food was terrific"}'
```

Use a specific model:

```bash
# Transformer Encoder (recommended)
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"text": "the food was terrific", "model": "encoder"}'

# Encoder-Decoder Transformer
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"text": "the food was terrific", "model": "decoder"}'
```

Response:

```json
{
  "text": "the food was terrific",
  "model": "encoder",
  "sentiment_score": 0.9701,
  "prediction": "positive"
}
```

List all available models:

```bash
curl http://localhost:8000/models
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
Score:      0.9701
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
| `--pos-weight` | `0.0` | Loss weight for positive class (0 = auto-calculate) |

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

All tuning and validation outputs include comprehensive classification metrics via [`sentimentizer/metrics.py`](sentimentizer/metrics.py):

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy (correct / total) |
| `positive_accuracy` | Accuracy on positive samples only (TP / (TP + FN)) |
| `negative_accuracy` | Accuracy on negative samples only (TN / (TN + FP)) |
| `precision` | Positive-class precision (TP / (TP + FP)) |
| `recall` | Positive-class recall (TP / (TP + FN)) |
| `f1` | Positive-class F1 score (harmonic mean of precision and recall) |
| `cohen_kappa` | Cohen's kappa coefficient (agreement beyond chance, -1 to 1) |
| `auc_roc` | Area under the ROC curve (requires probability scores) |
| `confusion_matrix` | TP, TN, FP, FN counts |

These metrics are computed in three places:

- **Ray Tune trials** — reported per epoch during hyperparameter search
- **Tuning Skill validation** — computed from known sentiment examples after model training
- **Programmatic API** — available via [`compute_metrics_from_model()`](sentimentizer/metrics.py) and [`compute_metrics_from_examples()`](sentimentizer/metrics.py)

```python
from sentimentizer.metrics import compute_metrics_from_model, compute_metrics_from_examples

# From a model and dataloader
metrics = compute_metrics_from_model(model, val_loader, device="cpu")
print(f"Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1:.4f}, Kappa: {metrics.cohen_kappa:.4f}")

# From validation result dicts
metrics = compute_metrics_from_examples(validation_results)
print(f"Positive accuracy: {metrics.positive_accuracy:.4f}")
print(f"Negative accuracy: {metrics.negative_accuracy:.4f}")
```

## Architecture

The pipeline consists of three stages, all powered by Ray:

1. **Extract** — Reads raw JSON data from `.zip` or `.tar` archives using `ray.data` and tokenizes text
2. **Transform** — Converts tokens to numeric sequences using `ray.data.map_batches()` and writes processed parquet
3. **Train** — Fits the model using either single-node PyTorch or distributed Ray Train with `TorchTrainer`

Inference is served via Ray Serve (see `serve.py` and `sentimentizer/serve.py`).

## Docker

Build and run the containerized service:

```bash
# Build
docker build -t sentimentizer .

# Run
docker run -p 8000:8000 -p 8265:8265 sentimentizer
```

The image uses a multi-stage build with Python 3.11-slim and CPU-only PyTorch. Port 8000 serves predictions; port 8265 exposes the Ray dashboard.

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

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (local-only mode)
uv sync

# Install with Ray distributed features
uv sync --extra ray

# Install with dev dependencies
uv sync --extra dev --extra ray
```

### With conda

```bash
conda create -n sentimentizer
conda install pip
pip install -e .
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
├── config.py            # Configuration dataclasses and constants
├── extractor.py         # Ray Data extraction from zip/tar archives
├── loader.py            # Data loading utilities
├── metrics.py           # Classification metrics (accuracy, F1, Cohen's kappa, AUC-ROC)
├── tokenizer.py         # Text tokenizer with pre-trained support
├── trainer.py           # Training logic
├── tuner.py             # Ray Tune + Optuna hyperparameter search
├── serve.py             # Ray Serve deployment app
├── hf.py                # Hugging Face Hub push/pull + model card generation
├── data/                # Training data (Yelp, GloVe)
├── agent/               # LLM-guided tuning agent
│   ├── __init__.py      # Package exports
│   ├── config.yaml      # Agent + tuner configuration (YAML)
│   ├── loader.py        # YAML → dataclass config loader
│   ├── models.py        # Pydantic models (AnalysisResult, TuningDecision, etc.)
│   ├── agents.py        # Pydantic AI agents (GLM 5.1 via Ollama)
│   ├── prompts.py       # System prompts for analysis & strategy agents
│   ├── state.py         # LangGraph AgentState TypedDict
│   ├── nodes.py         # LangGraph node functions (analyze, decide, tune, evaluate)
│   ├── graph.py         # LangGraph StateGraph + run_agent_tuning() entry point
│   └── skill.py         # TuningRun skill (tune → train → validate → retry pipeline)
└── models/
    ├── __init__.py
    ├── rnn.py           # RNN model with GloVe embeddings
    ├── encoder.py       # Transformer encoder model
    └── decoder.py       # Transformer decoder model
```

## License

[MIT](LICENSE)