## Complete Implementation Plan: ONNX Export + SetFit Router Pipeline

> **Status: IMPLEMENTED** — All code from this plan has been implemented and tested.
> See `AGENTS.md` for current conventions and `tests/test_export_onnx.py` / `tests/test_router.py` for test coverage.
>
> **Deviation from plan**: `optimum-onnx[onnxruntime]` was removed from `[onnx]` deps due to
> conflict with `numpy>=2.4.0`. The ONNX export uses `onnxruntime` directly via
> `onnxruntime.quantization.quantize_dynamic`. `requests` was added to `[router]` deps
> for the Ollama augmentation client.

---

### 1. Architecture Overview

**Goal**: Add ONNX export for all 3 existing models (RNN, Encoder, Decoder) and a SetFit-based routing classifier (3 categories: Dietary, Service, General). No Go router — Python inference only for v1.

**Models**:
| Model | Input | Output | Export Method |
|-------|-------|--------|---------------|
| RNN | Token IDs `(B, 200)` | Logits `(B, 3)` | `torch.onnx.export` via `_RNNOnnxWrapper` (masked fallback) |
| Encoder | Token IDs `(B, 200)` | Logits `(B, 3)` | `torch.onnx.export` (padding mask derived internally from input) |
| Decoder | Token IDs `(B, 200)` | Logits `(B, 3)` | `torch.onnx.export` (padding mask derived internally from input) |
| Router (SetFit) | Raw text string | Class probability `(3,)` | Python `setfit` inference (v1); ONNX deferred to v2 |

**Categories** (v1):
| Label | Category | Description |
|-------|----------|-------------|
| 0 | Dietary | Food allergies, celiac, FODMAP, ingredient safety |
| 1 | Service | Wait times, staff behavior, reservation issues |
| 2 | General | Ambiance, price, general food quality |

---

### 2. Directory Structure

```text
sentimentizer/
  export_onnx.py          # NEW: Unified ONNX export + quantization + validation
  router/                 # NEW: SetFit router module
    __init__.py           # Exports: SetFitConfig, RouteLabels, SEED_UTTERANCES
    config.py             # SetFitConfig, RouteLabels dataclasses
    seeds.py              # Golden example utterances per category
    augment.py            # GLM 5.1 augmentation via Ollama API
    dataset.py            # JSONL dataset loader, train/test split
    train_router.py        # SetFit training script
    evaluate.py           # Validation: similarity heatmap, threshold calibration
  models/
    rnn.py                # MODIFIED: add onnx_export flag to forward()
    encoder.py            # UNCHANGED
    decoder.py            # UNCHANGED
workflows/
  stages/
    export.py             # NEW: CLI workflow stage for ONNX export
  cli.py                  # MODIFIED: add export + router CLI commands
tests/
  test_export_onnx.py     # NEW: ONNX export + inference validation tests
  test_router.py          # NEW: SetFit router pipeline tests
onnx_artifacts/           # NEW: Gitignored output directory (auto-created by export)
  rnn.onnx
  rnn_quantized.onnx
  rnn_metadata.json
  encoder.onnx
  encoder_quantized.onnx
  encoder_metadata.json
  decoder.onnx
  decoder_quantized.onnx
  decoder_metadata.json
```

---

### 3. `sentimentizer/export_onnx.py` — Unified ONNX Export

```python
"""Export trained sentiment models to ONNX format with INT8 quantization."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sentimentizer import new_logger
from sentimentizer.config import (
    FileConfig,
    RNNConfig,
    EncoderConfig,
    DecoderConfig,
    weights_path_for,
)
from sentimentizer.models.rnn import RNN
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.decoder import Decoder

logger = new_logger(__name__)

ONNX_OPSET_VERSION = 17  # stable, well-tested; opset 18+ requires dynamo_export


class _RNNOnnxWrapper(nn.Module):
    """Wraps RNN to call forward(onnx_export=True) for ONNX tracing.

    torch.onnx.export(model, args, f) calls model(*args) internally.
    We cannot pass onnx_export=True as a keyword argument through the
    standard export call, so this wrapper forces the ONNX-compatible
    forward path (skipping pack_padded_sequence).
    """

    def __init__(self, rnn: RNN) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rnn.forward(inputs, onnx_export=True)


def load_model_for_export(model_type: str, device: str = "cpu") -> nn.Module:
    """Load a trained model for ONNX export (on CPU, in eval mode).

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        device: Device to load weights onto (use 'cpu' for ONNX export).

    Returns:
        Model in eval mode with dropout disabled.

    Raises:
        FileNotFoundError: If trained weights are not found locally or on HF Hub.
    """
    ...

def export_model_to_onnx(
    model: nn.Module,
    model_type: str,
    output_path: Path,
    seq_len: int = 200,
    opset_version: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export a trained sentiment model to ONNX format.

    For RNN models, wraps in _RNNOnnxWrapper to bypass pack_padded_sequence.

    Handles:
    - Dynamic batch and sequence length axes
    - RNN masked fallback (via _RNNOnnxWrapper)
    - Encoder/Decoder padding masks (derived internally from input == 0)

    Args:
        model: Trained model in eval mode.
        model_type: One of 'rnn', 'encoder', 'decoder'.
        output_path: Path to write the ONNX model file.
        seq_len: Maximum sequence length for the dummy input.
        opset_version: ONNX opset version (17 recommended).

    Returns:
        Path to the exported ONNX model file.
    """
    ...

def quantize_onnx_model(
    input_path: Path,
    output_path: Path,
) -> Path:
    """Apply INT8 dynamic quantization for CPU deployment (AVX-512 optimized).

    Uses onnxruntime.quantization.quantize_dynamic which quantizes weights
    to INT8 while keeping activations in FP32 — optimal for Zen 5 AVX-512.

    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Path to write the quantized ONNX model.

    Returns:
        Path to the quantized ONNX model file.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ...

def validate_onnx_export(
    onnx_path: Path,
    model: nn.Module,
    model_type: str,
    test_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> dict:
    """Verify ONNX model outputs match PyTorch within tolerance.

    For RNN models, the tolerance is relaxed to 1e-2 because the ONNX
    path (masked fallback) produces slightly different numerics than the
    pack_padded_sequence path. Encoder/Decoder models use 1e-4.

    Args:
        onnx_path: Path to the ONNX model file.
        model: Original PyTorch model (used for comparison).
        model_type: One of 'rnn', 'encoder', 'decoder'.
        test_input: Tensor to run through both models.
        tolerance: Maximum absolute difference allowed.

    Returns:
        Dict with 'max_diff', 'mean_diff', 'passed' keys.
    """
    ...

def export_pipeline(
    model_type: str,
    output_dir: Path = Path("onnx_artifacts"),
    quantize: bool = True,
    device: str = "cpu",
) -> dict:
    """Full export pipeline: load → export → quantize → validate.

    Creates output_dir if it doesn't exist. Saves metadata JSON alongside
    each ONNX model with model_type, opset_version, input_shape, dictionary
    path, and quantization status.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        output_dir: Directory for ONNX artifacts (auto-created).
        quantize: Whether to apply INT8 quantization.
        device: Device to load model onto (use 'cpu' for export).

    Returns:
        Dict with paths to all generated artifacts and validation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ...
```

**Key design decisions**:
- `ONNX_OPSET_VERSION = 17` — stable, supported by `torch.onnx.export` legacy exporter; opset 18+ requires `dynamo_export` (still preview)
- `_RNNOnnxWrapper` — wraps RNN to force `onnx_export=True` during `torch.onnx.export` tracing, since `torch.onnx.export()` calls `model(*args)` internally and cannot pass keyword arguments
- `load_model_for_export()` moves model to CPU, sets `eval()` mode, disables dropout
- `export_model_to_onnx()` uses `dynamic_axes` for batch_size and seq_len
- `validate_onnx_export()` runs the same input through both PyTorch and ONNX Runtime, compares outputs within tolerance (1e-2 for RNN, 1e-4 for Encoder/Decoder)
- Metadata JSON saved alongside each `.onnx` file with model_type, opset_version, input_shape, gensim dictionary path, and quantization info
- Encoder and Decoder derive their `src_key_padding_mask` from `inputs == 0` internally — no separate ONNX input is needed for the mask

---

### 4. RNN `forward()` Modification — Masked Fallback

The RNN's `forward()` currently uses `pack_padded_sequence`, which is incompatible with ONNX export (PyTorch issues #44988, #44971, #54261, #61154). Add an `onnx_export` parameter:

```python
# In sentimentizer/models/rnn.py

def forward(self, inputs: torch.Tensor, onnx_export: bool = False) -> torch.Tensor:
    """Forward pass producing raw logits.

    Args:
        inputs: Token IDs of shape (batch, seq_len)
        onnx_export: If True, skip pack_padded_sequence for ONNX compatibility.
                     Uses masked LSTM output instead. Slightly different numerics.
                     Do NOT pass this manually — it is set by _RNNOnnxWrapper
                     during torch.onnx.export() tracing.
    """
    embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
    embeds = self.dropout_layer(embeds)
    embeds = F.relu(self.fc0(embeds), inplace=True)

    lengths = (inputs != 0).sum(dim=1).clamp(min=1)  # (B,)

    if onnx_export:
        # ONNX-compatible path: no pack_padded_sequence.
        # The LSTM processes all positions including padding. Since padding
        # tokens map to the zero-vector in the embedding layer (index 0),
        # their contribution is limited to bias-driven drift in the hidden
        # state. For sequences near max_len=200, this drift is typically <1e-2.
        lstm_out, _ = self.lstm(embeds)  # (B, seq_len, hidden_size * 2)
        B = inputs.size(0)
        hidden_size = self.lstm.hidden_size

        # Forward final state: extract at the index of the last real token
        idx = (lengths - 1).clamp(max=inputs.size(1) - 1)
        forward_hidden = lstm_out[torch.arange(B, device=inputs.device), idx, :hidden_size]

        # Backward final state: at index 0 of the output.
        # NOTE: In the unpacked path, the backward LSTM at position 0 has
        # processed ALL tokens including padding. Since padding maps to
        # near-zero embeddings, the bias-driven drift is typically <1e-2
        # for sequences near max_len=200. validate_onnx_export() verifies
        # this tolerance.
        backward_hidden = lstm_out[:, 0, hidden_size:]

        hidden_cat = torch.cat([forward_hidden, backward_hidden], dim=1)
    else:
        # Standard path with packed sequences (more accurate)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

    logits = self.classifier(hidden_cat)  # (B, 1)
    return torch.squeeze(logits)  # (B,)
```

The `predict()` method and training are **unchanged** — they always use the packed path (default `onnx_export=False`). Only `_RNNOnnxWrapper.forward()` calls `self.rnn.forward(inputs, onnx_export=True)` during ONNX export tracing.

---

### 5. `sentimentizer/router/` — SetFit Router Module

#### `__init__.py`

```python
"""SetFit router module for Yelp review categorization."""

from sentimentizer.router.config import SetFitConfig, RouteLabels
from sentimentizer.router.seeds import SEED_UTTERANCES

__all__ = ["SetFitConfig", "RouteLabels", "SEED_UTTERANCES"]
```

#### `config.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SetFitConfig:
    """Configuration for the SetFit router training pipeline.

    Default base model is BAAI/bge-base-en-v1.5 (109M params, 768-dim embeddings, strong MTEB scores) to match
    the project's "smole" philosophy. Upgrade to mxbai-embed-large-v1
    (335M params, ~1.3GB) only if evaluation fails to meet thresholds.
    """
    base_model: str = "BAAI/bge-base-en-v1.5"
    num_iterations: int = 20      # contrastive pairs per example
    num_epochs: int = 1           # fine-tuning epochs
    batch_size: int = 16
    max_seq_length: int = 512
    seed: int = 42
    output_dir: Path = Path("models/router")

@dataclass(frozen=True)
class RouteLabels:
    """Route category labels for the Yelp review classifier."""
    dietary: int = 0
    service: int = 1
    general: int = 2

    @classmethod
    def label_names(cls) -> dict[int, str]:
        return {0: "dietary", 1: "service", 2: "general"}

    @classmethod
    def num_classes(cls) -> int:
        return 3
```

#### `seeds.py`

5–10 golden examples per category stored as `list[dict[str, str | int]]`:

```python
SEED_UTTERANCES: list[dict[str, str | int]] = [
    # Dietary (label 0)
    {"text": "They were so careful with my celiac needs.", "label": 0},
    {"text": "I asked if the soup had gluten and the chef came out to tell me.", "label": 0},
    {"text": "My nut allergy was taken seriously here, they even cleaned the grill.", "label": 0},
    {"text": "The menu clearly marked all dairy-free options.", "label": 0},
    {"text": "I have a shellfish allergy and they prepared my dish separately.", "label": 0},
    {"text": "They substituted tamari for soy sauce for my soy allergy.", "label": 0},
    {"text": "The server double-checked with the kitchen about cross-contamination.", "label": 0},
    {"text": "As a vegan I felt completely safe eating here.", "label": 0},
    {"text": "They accidentally served me regular bread instead of gluten-free.", "label": 0},
    {"text": "The kitchen uses shared fryers, so it's not safe for celiacs.", "label": 0},

    # Service (label 1)
    {"text": "The waiter brought me the wrong order.", "label": 1},
    {"text": "We waited 45 minutes for a table even with a reservation.", "label": 1},
    {"text": "The host was incredibly rude when we arrived.", "label": 1},
    {"text": "Our server checked on us constantly and refilled our drinks.", "label": 1},
    {"text": "They forgot our appetizer and didn't apologize.", "label": 1},
    {"text": "The manager comped our meal after we complained about the wait.", "label": 1},
    {"text": "Service was incredibly slow even though the restaurant was empty.", "label": 1},
    {"text": "The bartender was attentive and remembered our names.", "label": 1},
    {"text": "They refused to seat us even though we had a confirmed booking.", "label": 1},
    {"text": "Our food came out cold and the server didn't offer to reheat it.", "label": 1},

    # General (label 2)
    {"text": "The garlic bread was way too salty.", "label": 2},
    {"text": "Great ambiance but the prices are steep for what you get.", "label": 2},
    {"text": "The decor is beautiful and the music is just right.", "label": 2},
    {"text": "Portions are huge but the quality is just okay.", "label": 2},
    {"text": "This place has an amazing view of the city skyline.", "label": 2},
    {"text": "The pasta was decent but nothing special.", "label": 2},
    {"text": "Best pizza I've had in this neighborhood.", "label": 2},
    {"text": "The restaurant is cozy but the food is overpriced.", "label": 2},
    {"text": "Loud music made it hard to have a conversation.", "label": 2},
    {"text": "The dessert menu is worth staying for.", "label": 2},
]
```

#### `augment.py`

Ollama API client for GLM 5.1 augmentation with hard-negative generation:

```python
"""Augment seed utterances using GLM 5.1 via Ollama API.

Generates hard negatives: text that sounds like one category but
belongs to another (e.g., "The bread was so dry" → General, not Dietary).
"""

import json
import time
from typing import Optional

import requests

from sentimentizer import new_logger
from sentimentizer.router.config import RouteLabels

logger = new_logger(__name__)

LABEL_NAMES = RouteLabels.label_names()


def _build_prompt(seed_text: str, seed_label: int, num_variations: int) -> str:
    """Build the augmentation prompt for GLM 5.1."""
    seed_category = LABEL_NAMES[seed_label]
    other_categories = [v for k, v in LABEL_NAMES.items() if k != seed_label]
    return (
        f"Generate {num_variations} variations of this Yelp review: "
        f'"{seed_text}"\n\n'
        f"The original is categorized as '{seed_category}'. "
        f"Generate variations that:\n"
        f"1. Stay in the '{seed_category}' category (at least 60%)\n"
        f"2. Sound like they belong to '{other_categories[0]}' or "
        f"'{other_categories[1]}' but are actually '{seed_category}' (hard negatives, at least 20%)\n\n"
        f"Output as JSONL with fields: text, label (where label={seed_label})\n"
        f"One JSON object per line."
    )


def augment_seeds(
    seeds: list[dict],
    model: str = "glm5.1",
    ollama_url: str = "http://localhost:11434/api/generate",
    variations_per_seed: int = 50,
    batch_size: int = 5,
) -> list[dict]:
    """Expand seed utterances using GLM 5.1 via Ollama API.

    Args:
        seeds: List of {"text": str, "label": int} seed utterances.
        model: Ollama model name.
        ollama_url: Ollama API endpoint.
        variations_per_seed: Number of variations to generate per seed.
        batch_size: Number of seeds to process in one API call.

    Returns:
        Original seeds + augmented utterances with the same format.
    """
    augmented = list(seeds)  # start with originals

    for i in range(0, len(seeds), batch_size):
        batch = seeds[i : i + batch_size]
        for seed in batch:
            prompt = _build_prompt(seed["text"], seed["label"], variations_per_seed)
            try:
                response = requests.post(
                    ollama_url,
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                text = result.get("response", "")
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "text" in entry and "label" in entry:
                            augmented.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse augmentation line: {line}")
            except requests.RequestException as e:
                logger.error(f"Ollama API request failed: {e}")
            time.sleep(0.5)  # rate limiting

    logger.info(f"Augmented {len(seeds)} seeds → {len(augmented)} total utterances")
    return augmented
```

#### `dataset.py`

```python
"""Dataset loading and train/test split for the router."""

from pathlib import Path
from typing import Optional

from datasets import load_dataset

from sentimentizer import new_logger

logger = new_logger(__name__)


def load_router_dataset(
    data_path: str = "augmented_yelp.jsonl",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Load JSONL dataset and split into train/test.

    Format: {"text": "...", "label": 0|1|2}

    Args:
        data_path: Path to the JSONL file.
        test_size: Fraction of data for test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    dataset = load_dataset("json", data_files=data_path)
    split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    logger.info(
        f"Loaded router dataset: {len(split['train'])} train, "
        f"{len(split['test'])} test"
    )
    return split["train"], split["test"]
```

#### `train_router.py`

```python
"""Train the SetFit router model for Yelp review categorization."""

from pathlib import Path

from setfit import SetFitModel, Trainer, TrainingArguments

from sentimentizer import new_logger
from sentimentizer.router.config import SetFitConfig

logger = new_logger(__name__)


def train_router(
    config: SetFitConfig,
    train_dataset,
    eval_dataset=None,
) -> SetFitModel:
    """Train the SetFit router model.

    Uses CosineSimilarityLoss for contrastive learning on the embedding
    space, then fits a classification head on top of the embeddings.

    Args:
        config: SetFitConfig with model parameters.
        train_dataset: Training dataset with 'text' and 'label' columns.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Trained SetFitModel.
    """
    model = SetFitModel.from_pretrained(config.base_model)

    args = TrainingArguments(
        batch_size=config.batch_size,
        num_iterations=config.num_iterations,
        num_epochs=config.num_epochs,
        seed=config.seed,
        output_dir=str(config.output_dir),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()
    model.save_pretrained(str(config.output_dir))
    logger.info(f"Router model saved to {config.output_dir}")
    return model
```

#### `evaluate.py`

```python
"""Evaluation utilities for the SetFit router model.

Provides:
- Cosine similarity matrix (inter-class < 0.65, intra-class > 0.85)
- Tau threshold calibration (where False Positives appear in General category)
- Full evaluation: accuracy, F1, similarity matrix, threshold
"""

import numpy as np
from sklearn.metrics import classification_report

from sentimentizer import new_logger
from sentimentizer.router.config import RouteLabels

logger = new_logger(__name__)


def compute_similarity_matrix(model, texts_by_label: dict[int, list[str]]) -> np.ndarray:
    """Generate inter-class and intra-class cosine similarity heatmap.

    Args:
        model: Trained SetFitModel.
        texts_by_label: Dict mapping label int to list of example texts.

    Returns:
        Cosine similarity matrix of shape (num_classes, num_classes).
    """
    ...


def calibrate_threshold(model, eval_dataset) -> float:
    """Find tau threshold where False Positives appear in General category.

    The tau threshold is the confidence score below which the router
    should fall back to a default handling strategy.

    Returns:
        Recommended tau threshold.
    """
    ...


def evaluate_router(model, eval_dataset) -> dict:
    """Full evaluation: accuracy, F1, similarity matrix, threshold.

    Returns:
        Dict with 'accuracy', 'f1', 'classification_report',
        'similarity_matrix', 'tau_threshold'.
    """
    ...
```

---

### 6. `workflows/stages/export.py` — CLI Stage

```python
"""ONNX export workflow stage."""

import logging
from pathlib import Path

from sentimentizer.export_onnx import export_pipeline

logger = logging.getLogger(__name__)


def run_export(
    state,
    model_type: str,
    quantize: bool = True,
    output_dir: str = "onnx_artifacts",
) -> None:
    """Export a trained model to ONNX format.

    Args:
        state: State object with model type info (unused, for CLI consistency).
        model_type: One of 'rnn', 'encoder', 'decoder'.
        quantize: Whether to apply INT8 quantization.
        output_dir: Directory for ONNX artifacts.
    """
    results = export_pipeline(
        model_type=model_type,
        output_dir=Path(output_dir),
        quantize=quantize,
    )
    logger.info(f"Export complete: {results}")
```

---

### 7. CLI Changes — `workflows/cli.py`

Add `export` command and `router` command group:

```python
# ── export ─────────────────────────────────────

@cli.command()
@click.option(
    "--model",
    type=click.Choice(["rnn", "encoder", "decoder"]),
    required=True,
    help="Model to export to ONNX",
)
@click.option("--quantize/--no-quantize", default=True, help="Apply INT8 quantization")
@click.option("--output-dir", default="onnx_artifacts", help="Output directory")
@click.pass_context
def export(ctx: click.Context, model: str, quantize: bool, output_dir: str) -> None:
    """Export a trained model to ONNX format."""
    from workflows.stages.export import run_export
    run_export(ctx.obj, model_type=model, quantize=quantize, output_dir=output_dir)


# ── router ─────────────────────────────────────

@cli.group()
def router() -> None:
    """SetFit router operations."""


@router.command("train")
@click.option("--data", type=click.Path(exists=True), help="Path to augmented JSONL data")
@click.option("--base-model", default=None, help="SetFit base model (overrides config default)")
@click.option("--output-dir", default="models/router", help="Output directory for trained model")
@click.pass_context
def router_train(ctx: click.Context, data: str, base_model: str | None, output_dir: str) -> None:
    """Train the SetFit router model."""
    from sentimentizer.router.config import SetFitConfig
    from sentimentizer.router.dataset import load_router_dataset
    from sentimentizer.router.train_router import train_router

    config = SetFitConfig(
        base_model=base_model or SetFitConfig.base_model,
        output_dir=Path(output_dir),
    )
    train_ds, eval_ds = load_router_dataset(data)
    model = train_router(config, train_ds, eval_ds)
    ...


@router.command("evaluate")
@click.option("--model-path", required=True, help="Path to trained router model")
@click.pass_context
def router_evaluate(ctx: click.Context, model_path: str) -> None:
    """Evaluate the SetFit router model."""
    from sentimentizer.router.evaluate import evaluate_router
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(model_path)
    ...
```

---

### 8. Dependencies — `pyproject.toml`

```toml
[project.optional-dependencies]
router = [
    "setfit>=1.1.0",
    "datasets",
]
onnx = [
    "onnx",
    "onnxruntime",
    "optimum-onnx[onnxruntime]",  # optimum v2.0+ moved ONNX to this package
]
```

Users install with:
```bash
pip install -e ".[router]"    # For SetFit training
pip install -e ".[onnx]"      # For ONNX export
pip install -e ".[router,onnx]"  # Both
```

**Note**: `optimum-onnx[onnxruntime]` is the correct package for `optimum` v2.0+ (released Oct 2025). The old `optimum[onnxruntime]` package no longer includes ONNX export functionality.

---

### 9. `.gitignore` Update

Add:
```text
# ONNX export artifacts
onnx_artifacts/

# SetFit router model output
models/router/
```

---

### 10. Testing Plan

**`tests/test_export_onnx.py`**:
- `TestExportRNN`: Export RNN to ONNX, validate outputs match PyTorch (tolerance 1e-2)
- `TestExportEncoder`: Export Encoder to ONNX, validate outputs match PyTorch (tolerance 1e-4)
- `TestExportDecoder`: Export Decoder to ONNX, validate outputs match PyTorch (tolerance 1e-4)
- `TestQuantization`: Verify INT8 quantized model produces outputs within tolerance
- `TestRNNOnnxExportMode`: Verify `_RNNOnnxWrapper` produces different (but valid) outputs vs packed path
- `TestExportPipeline`: Full pipeline test — load → export → quantize → validate
- All tests use `@pytest.mark.skipif(not ONNX_AVAILABLE)` to skip in CI without `onnxruntime`

**`tests/test_router.py`**:
- `TestSetFitConfig`: Verify config defaults, frozen dataclass
- `TestRouteLabels`: Verify label mapping, `num_classes()`
- `TestSeedUtterances`: Verify seed data integrity (correct labels 0/1/2, non-empty text, 10 per category)
- `TestDatasetLoader`: Verify JSONL loading and train/test split (with mock data)
- `TestAugmentSeeds`: Mock Ollama API, verify augmentation output format
- `TestRNNOnnxWrapper`: Verify wrapper calls `forward(inputs, onnx_export=True)`

---

### 11. Key Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ONNX opset | 17 | Stable, well-tested; opset 18+ requires `dynamo_export` (still preview) |
| RNN export | `_RNNOnnxWrapper` with masked fallback | `pack_padded_sequence` is ONNX-incompatible; wrapper avoids kwarg issue with `torch.onnx.export` |
| RNN tolerance | 1e-2 | Masked fallback has slight numeric drift from padding; Encoder/Decoder use 1e-4 |
| Router export | Python `setfit` inference (v1) | SetFit models need separate sentence-transformer + head export; deferred to v2 |
| Router base model | `BAAI/bge-base-en-v1.5` (default) | 109M params, 768-dim embeddings, strong MTEB scores; upgrade to `mxbai-embed-large-v1` only if eval fails |
| Quantization | INT8 dynamic (`quantize_dynamic`) | Optimal for AVX-512, keeps FP32 activations |
| SetFit version | `>=1.1.0` | Uses Sentence Transformers backend; old `SetFitTrainer` deprecated |
| Optimum ONNX | `optimum-onnx[onnxruntime]` | `optimum` v2.0+ moved ONNX functionality to separate package |
| Output dir | `onnx_artifacts/` | Avoids collision with `sentimentizer/models/` (Python source) |
| Router module | `sentimentizer/router/` | Avoids shadowing the `setfit` library |

---

### 12. Implementation Order

1. **`pyproject.toml`** — Add `router` and `onnx` optional dependency groups
2. **`.gitignore`** — Add `onnx_artifacts/` and `models/router/`
3. **`sentimentizer/models/rnn.py`** — Add `onnx_export` flag to `forward()`
4. **`sentimentizer/export_onnx.py`** — Unified export, quantization, validation (with `_RNNOnnxWrapper`)
5. **`sentimentizer/router/`** — Config, seeds, augment, dataset, train, evaluate
6. **`workflows/stages/export.py`** — CLI export stage
7. **`workflows/cli.py`** — Add `export` command and `router` group
8. **`tests/test_export_onnx.py`** — ONNX export tests
9. **`tests/test_router.py`** — Router pipeline tests
10. **`AGENTS.md`** — Document new conventions

---

### 13. Risks & Mitigations

#### A. RNN Fallback Numerics (Padding Noise)
- **Risk**: Without `pack_padded_sequence`, the backward LSTM accumulates state changes when processing padding tokens, due to non-zero LSTM biases. This causes the hidden state to drift.
- **Mitigation**: The `_RNNOnnxWrapper` extracts the backward hidden state from `lstm_out[:, 0, hidden_size:]`, which includes padding influence. For sequences near `max_len=200`, the drift is typically <1e-2. `validate_onnx_export()` uses tolerance 1e-2 for RNN models to account for this.

#### B. Dependency Conflicts (`torch` vs. `setfit`)
- **Risk**: The project uses a recent `torch` version. `setfit>=1.1.0` depends on `sentence-transformers`, which may enforce strict upper bounds on `torch`.
- **Mitigation**: Test `pip install -e ".[router,onnx]"` resolves cleanly. If needed, use `--no-deps` or pin specific versions. Add a CI step that verifies imports.

#### C. Router Model Footprint
- **Risk**: `mxbai-embed-large-v1` (335M params, ~1.3GB) contradicts the project's "smole" philosophy and introduces severe cold-start latencies.
- **Mitigation**: Default to `BAAI/bge-base-en-v1.5` (109M params, 768-dim embeddings, strong MTEB scores). Only upgrade to `mxbai-embed-large-v1` if evaluation fails to meet inter-class < 0.65 / intra-class > 0.85 thresholds.

#### D. Architectural Integration (Go vs. Python)
- **Risk**: If the primary web service is written in Go, forcing it to call a Python process for SetFit routing introduces IPC/Network latency, negating ONNX's sub-millisecond execution speeds.
- **Mitigation**: Prioritize exporting the SetFit router (sentence-transformer + classification head) to ONNX earlier than v2, allowing the entire pipeline to run natively in Go via ONNX Runtime without Python overhead.