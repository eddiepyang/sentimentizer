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

- 🚀 **[Model Serving Guide](docs/serving.md)**: Ray Serve application deployment, FastAPI endpoints (sentiment/routing), and the Go CLI client.
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
├── serve/                 # Ray Serve deployment: FastAPI + @serve.ingress, /v1/ prefix
│   ├── app.py             # FastAPI route handlers and deployment class
│   ├── base.py            # ServiceMetrics (request/latency tracking), _DummyServe fallback
│   ├── config.py           # Serve deployment configuration (YAML/env var loading, incl. cors_origins)
│   └── models.py          # Pydantic request/response models for Swagger docs
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
