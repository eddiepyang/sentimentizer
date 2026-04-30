# Introduction

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
  
Beta release, api subject to change. Install with:  

```
pip install sentimentizer
```  
  
This repo contains Neural Nets written with the pytorch framework for sentiment analysis. 
Small models can be pretty effective for classification tasks at a much smaller cost to deploy.
This package focuses on sentiment analysis and all models were trained on a single 2080Ti gpu in minutes. 
Deploying models for inference requires less than 1GB of memory which makes creating multiple containers relatively efficient.


## Usage
```
# where 0 is very negative and 1 is very positive
from sentimentizer.tokenizer import get_trained_tokenizer
from sentimentizer.models.rnn import get_trained_model

model = get_trained_model(64, 'cpu')
tokenizer = get_trained_tokenizer()
review_text = "greatest pie ever, best in town!"
positive_ids = tokenizer.tokenize_text(review_text)
model.predict(positive_ids)
  
>> tensor(0.9701)
```

## Install for development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
```

Or with miniconda:
```
conda create -n {env}  
conda install pip  
pip install -e .  
```

## Retrain model
To rerun the model:
* get the yelp [dataset](https://www.yelp.com/dataset), 
* get the glove 6B 100D [dataset](https://nlp.stanford.edu/projects/glove/)
* place both files in the package data directory 
* run the training script in workflows

### Single-node training
```bash
python workflows/driver.py --device cuda --type new --save True
```

### Distributed training with Ray Train
```bash
# Run with 2 workers (default)
python workflows/driver.py --device cuda --distributed --save True

# Run with 4 workers
python workflows/driver.py --device cuda --distributed --num-workers 4 --save True

# Run on CPU only
python workflows/driver.py --device cpu --distributed --num-workers 2
```

The `--distributed` flag enables Ray Train, which distributes data and model training
across multiple workers. Each worker gets a shard of the dataset and runs the training
loop with PyTorch Distributed Data Parallel (DDP). Checkpoints and metrics are
aggregated automatically by Ray Train.

### CLI arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | Device to use: `cuda`, `mps`, or `cpu` |
| `--model` | `rnn` | Model type: `rnn`, `encoder`, or `decoder` |
| `--type` | `new` | Run type: `new` (from scratch) or `update` (resume) |
| `--stop` | `10000` | Number of lines to load from the dataset |
| `--save` | `False` | Save model weights after training |
| `--distributed` | `False` | Enable distributed training with Ray Train |
| `--num-workers` | `2` | Number of Ray Train workers (distributed mode only) |

## Architecture

The pipeline consists of three stages, all powered by Ray:

1. **Extract** — Reads raw JSON data from zip archives using `ray.data` and tokenizes text
2. **Transform** — Converts tokens to numeric sequences using `ray.data.map_batches()` and writes processed parquet
3. **Train** — Fits the model using either single-node PyTorch or distributed Ray Train with `TorchTrainer`

Inference is served via Ray Serve (see `serve.py`).

## Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run only Ray Train tests
uv run pytest tests/ -v -k "Ray"

# Run with coverage
uv run pytest tests/ -v --cov=sentimentizer --cov-report=term-missing
