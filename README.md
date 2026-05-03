# sentimentizer

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight PyTorch models for sentiment analysis. Small models can be pretty effective for classification tasks at a much smaller cost to deploy — all models were trained on a single 2080Ti GPU in minutes, and inference requires less than 1GB of memory.

> **Beta release** — API is subject to change.

## Install
```bash
pip install sentimentizer
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

The `serve.py` entry point deploys a Ray Serve application that loads **all three models** (RNN, Encoder, Decoder) at startup. You can select which model to use per request via the `model` field.

```bash
serve run serve:app --host 0.0.0.0 --port 8000
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

Sentimentizer offers three ways to tune hyperparameters, each at a different level of automation:

| | **Standalone** | **Agent Tuning** | **Tuning Skill** |
|---|---|---|---|
| **What it does** | Single Ray Tune + Optuna search | LLM-guided iterative search loop | Full pipeline: tune → train → validate → retry |
| **LLM involved** | ❌ No | ✅ GLM 5.1 via Ollama | ✅ (in agent mode) or ❌ (in standalone mode) |
| **Iterative** | ❌ One-shot sweep | ✅ Refines search space each iteration | ✅ Refines + validates + retries |
| **Model validation** | ❌ | ❌ | ✅ Tests predictions on known examples |
| **Auto-retry on failure** | ❌ | ❌ | ✅ Re-tunes up to `max_retries` times |
| **Saves final model** | ❌ | ❌ | ✅ Trains & saves best model weights |
| **Requires Ollama** | ❌ No | ✅ Yes | Only in agent mode |
| **CLI flag** | `--tune --tune-mode standalone` | `--agent-tune` | `--tune` (defaults to agent mode) |
| **When to use** | Quick sweep, no Ollama available | You want LLM-guided search but will handle model training yourself | You want a complete end-to-end pipeline |

### Standalone Tuning

Runs a single Ray Tune + Optuna hyperparameter search with no LLM involvement. Best for quick sweeps or when Ollama is unavailable.

```bash
# Via Makefile
make tune-standalone

# Via CLI
python workflows/driver.py --model rnn --tune --tune-mode standalone --save
```

This executes one [`tune_model()`](sentimentizer/tuner.py) call — it searches the space defined in [`sentimentizer/agent/config.yaml`](sentimentizer/agent/config.yaml) and returns the best configuration found. No iterative refinement, no model validation.

#### Output

Returns a dict with the best configuration and metrics from the single search:

| Key | Description |
|-----|-------------|
| `best_config` | Best hyperparameter configuration found (e.g., `{"lr": 0.003, "hidden_size": 256}`) |
| `best_accuracy` | Best validation accuracy across all trials |
| `best_loss` | Best validation loss across all trials |
| `best_precision` | Best positive-class precision (TP / (TP + FP)) |
| `best_recall` | Best positive-class recall (TP / (TP + FN)) |
| `best_f1` | Best positive-class F1 score |
| `best_cohen_kappa` | Best Cohen's kappa coefficient |
| `best_positive_accuracy` | Best accuracy on positive samples |
| `best_negative_accuracy` | Best accuracy on negative samples |
| `trial_count` | Number of Ray Tune trials completed |

When run via the Tuning Skill (`--tune --tune-mode standalone`), this is wrapped with model training, validation, and retry logic (see below).

### Agent Tuning

An LLM-guided hyperparameter tuning loop that uses **Pydantic AI Slim** (GLM 5.1 via Ollama) for reasoning, **LangGraph** for workflow orchestration, and **Ray Tune + Optuna** for the search backend. The agent iteratively refines the search space based on results from previous iterations.

#### Architecture

```
analyze (GLM 5.1) → decide (GLM 5.1) → tune (Ray Tune + Optuna) → evaluate
     ↑                                                              │
     └──────────────────────────────────────────────────────────────┘
                          (loop until converged)
```

1. **analyze** — GLM 5.1 examines training metrics, detects overfitting/underfitting, assesses learning rate
2. **decide** — GLM 5.1 chooses a strategy (widen, narrow, change_focus, increase_epochs, stop) and produces a validated `TuningDecision` with an updated search space
3. **tune** — Ray Tune + Optuna executes the hyperparameter search with ASHA scheduling
4. **evaluate** — Checks convergence (improvement below threshold for 3 iterations, max iterations reached, or agent decides to stop)

#### Prerequisites

Install [Ollama](https://ollama.ai) and pull the GLM 5.1 model:

```bash
ollama pull glm5.1
```

#### Output

The agent returns an [`AgentRunResult`](sentimentizer/agent/models.py) with:

| Field | Description |
|-------|-------------|
| `best_config` | Best hyperparameter configuration found (e.g., `{"lr": 0.003, "hidden_size": 256}`) |
| `best_accuracy` | Best validation accuracy achieved across all iterations |
| `best_loss` | Best validation loss achieved |
| `iterations_completed` | Number of agent loop iterations that ran |
| `converged` | Whether the agent converged before reaching `max_iterations` |
| `history` | List of [`TuningResult`](sentimentizer/agent/models.py) from each iteration |

The result is always written to `best_config.json`:

```json
{
  "best_config": {"lr": 0.003, "hidden_size": 256, "num_layers": 2, "dropout": 0.2},
  "best_accuracy": 0.89,
  "best_loss": 0.31,
  "iterations": 3,
  "converged": true
}
```

> **Note:** Agent tuning (`--agent-tune`) only runs the LLM-guided search loop — it finds the best hyperparameters but does **not** train a final model or validate predictions. To get a trained, validated model, use the Tuning Skill below.

#### Usage

```bash
# Via Makefile
make train-agent

# Via CLI
python workflows/driver.py --model encoder --agent-tune --save

# With a custom agent config
python workflows/driver.py --model encoder --agent-tune --agent-config path/to/custom.yaml --save
```

### Tuning Skill

The **Tuning Skill** (`TuningRun` in [`sentimentizer/agent/skill.py`](sentimentizer/agent/skill.py)) is the highest-level tuning interface. It wraps either agent-guided or standalone tuning with additional post-tuning steps:

1. **Tune** — Runs agent-guided (`mode="agent"`) or standalone (`mode="standalone"`) hyperparameter search
2. **Train** — Trains a final model using the best configuration found (2× default epochs for better convergence)
3. **Validate** — Tests the trained model against known sentiment examples (e.g., "amazing food great service" → positive, "terrible experience" → negative)
4. **Retry** — If validation fails (accuracy below threshold), re-tunes with adjusted parameters up to `max_retries` times

```
┌──────────────────────────────────────────────────┐
│                Tuning Skill                       │
│                                                   │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐  │
│  │  Tune    │───▶│  Train  │───▶│  Validate    │  │
│  │(agent or │    │  final  │    │  predictions │  │
│  │standalone)│   │  model  │    │  on known    │  │
│  └─────────┘    └─────────┘    │  examples    │  │
│       ▲                        └──────┬───────┘  │
│       │                               │           │
│       └─────────── retry ────────────┘           │
│              (if validation fails)                 │
└──────────────────────────────────────────────────┘
```

#### Output

Returns a [`TuningRunResult`](sentimentizer/agent/skill.py) with:

| Field | Description |
|-------|-------------|
| `best_config` | Best hyperparameter configuration found |
| `best_accuracy` | Best validation accuracy achieved |
| `best_loss` | Best validation loss achieved |
| `best_precision` | Best positive-class precision (TP / (TP + FP)) |
| `best_recall` | Best positive-class recall (TP / (TP + FN)) |
| `best_f1` | Best positive-class F1 score |
| `best_cohen_kappa` | Best Cohen's kappa coefficient |
| `best_positive_accuracy` | Best accuracy on positive samples |
| `best_negative_accuracy` | Best accuracy on negative samples |
| `iterations_completed` | Number of tuning iterations (1 for standalone, variable for agent) |
| `converged` | Whether the agent converged before max iterations |
| `model_path` | Path to the saved model weights (`.pth` file) |
| `results_path` | Path to the saved JSON results file |
| `validation_passed` | Whether model predictions met the validation threshold |
| `validation_results` | Per-example validation details (text, expected, score, correct) |
| `validation_metrics` | Full [`ClassificationMetrics`](sentimentizer/metrics.py) dict from model validation |
| `retry_count` | Number of re-tuning attempts due to failed validation |
| `elapsed_seconds` | Wall-clock time for the entire run |

Results are saved to `tuning_results/tuning_results_{model_type}.json`. If validation passes, the best model weights are also copied to the default weights path for serving.

#### Usage

```bash
# Agent-guided tuning with model validation (recommended, defaults to RNN)
make tune

# Tune specific models
make tune-rnn
make tune-encoder
make tune-decoder

# Standalone mode (no LLM, single Ray Tune sweep, still validates model)
make tune-standalone

# Customize the number of trials and agent iterations
make tune-custom SAMPLES=50 ITERATIONS=10

# Skip model validation
make tune-no-validate
```

Via CLI:

```bash
# Agent-guided skill (default)
python workflows/driver.py --model rnn --tune --save

# Standalone skill (no LLM)
python workflows/driver.py --model rnn --tune --tune-mode standalone --save

# Customize trials, iterations, and validation
python workflows/driver.py --model encoder --tune --save \
  --tune-samples 50 \
  --tune-max-iterations 10 \
  --validation-threshold 0.8 \
  --max-retries 3

# Skip validation
python workflows/driver.py --model rnn --tune --no-validate --save
```

Programmatic API:

```python
from sentimentizer.agent.skill import TuningRun, TuningRunConfig

# Agent-guided tuning with validation (recommended)
config = TuningRunConfig(model_type="rnn", mode="agent")
result = TuningRun(config).execute()
print(f"Best accuracy: {result.best_accuracy:.4f}")
print(f"Validation passed: {result.validation_passed}")

# Standalone tuning with validation
config = TuningRunConfig(model_type="encoder", mode="standalone")
result = TuningRun(config).execute()

# Quick convenience function
from sentimentizer.agent.skill import create_tuning_run
result = create_tuning_run(model_type="rnn", mode="agent")
```

### Configuration

Agent and tuner settings are defined in [`sentimentizer/agent/config.yaml`](sentimentizer/agent/config.yaml):

```yaml
agent:
  model_name: glm5.1                    # Ollama model name
  ollama_base_url: http://localhost:11434/v1
  max_iterations: 5                      # Max agent loop iterations
  convergence_threshold: 0.005           # Stop if avg improvement < threshold over 3 iterations
  temperature: 0.3                       # LLM sampling temperature
  max_tokens: 2048                       # Max LLM output tokens
  checkpointing:
    enabled: true
    db_path: agent_checkpoints.db
  human_in_the_loop: false               # Require human approval (future)

tuner:
  scheduler: asha                        # asha, hyperband, or median
  metric: val_accuracy
  mode: max
  num_samples: 20                        # Trials per tuning iteration
  grace_period: 2
  reduction_factor: 3
  search_spaces:
    rnn:
      lr: { type: loguniform, low: 1e-5, high: 1e-2 }
      hidden_size: { type: choice, values: [128, 256, 512] }
      ...
```

Override the config path via the `SENTIMENTIZER_AGENT_CONFIG` environment variable.

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

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
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