.PHONY: setup setup-dev download-data train train-rnn train-encoder train-decoder \
       train-distributed train-quick serve test lint format clean docker-build docker-run

# Default device: use auto-detect (cuda > mps > cpu)
DEVICE ?= auto
# Default model type
MODEL ?= rnn
# Default number of lines to load
STOP ?= 10000
# Checkpoint directory (empty = no checkpointing)
CHECKPOINT_DIR ?=

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

## Install dependencies (production only)
setup:
	uv sync

## Install dependencies with dev tools (pytest, ruff, black, etc.)
setup-dev:
	uv sync --extra dev

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

## Download GloVe embeddings and print Yelp dataset instructions
download-data:
	bash scripts/download_data.sh

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

## Train a model (defaults: --model rnn --device auto --stop 10000)
train:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop $(STOP) --save

## Train RNN model
train-rnn:
	uv run python workflows/driver.py --device $(DEVICE) --model rnn --type new --stop $(STOP) --save

## Train Transformer Encoder model (recommended)
train-encoder:
	uv run python workflows/driver.py --device $(DEVICE) --model encoder --type new --stop $(STOP) --save

## Train Transformer Decoder model
train-decoder:
	uv run python workflows/driver.py --device $(DEVICE) --model decoder --type new --stop $(STOP) --save

## Quick training run with fewer rows for iteration
train-quick:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop 5000 --save

## Train with checkpointing enabled (saves to CHECKPOINT_DIR, defaults to checkpoints/)
train-checkpoint:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop $(STOP) \
		--checkpoint-dir $(or $(CHECKPOINT_DIR),checkpoints/) --checkpoint-every 1 --save

## Resume training from the latest checkpoint
train-resume:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop $(STOP) \
		--checkpoint-dir $(or $(CHECKPOINT_DIR),checkpoints/) --resume --save

## Distributed training with Ray Train (2 workers by default)
train-distributed:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop $(STOP) \
		--distributed --save

## Distributed training with custom worker count (usage: make train-dist-workers WORKERS=4)
train-dist-workers:
	uv run python workflows/driver.py --device $(DEVICE) --model $(MODEL) --type new --stop $(STOP) \
		--distributed --num-workers $(WORKERS) --save

## Agent-guided hyperparameter tuning (requires Ollama with glm5.1)
train-agent:
	uv run python workflows/driver.py --model $(MODEL) --agent-tune --save

# ──────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────

## Start Ray Serve with all three models
serve:
	serve run sentimentizer.serve:app --host 0.0.0.0 --port 8000

# ──────────────────────────────────────────────
# Testing & Linting
# ──────────────────────────────────────────────

## Run all tests with verbose output
test:
	uv run pytest tests/ -v

## Run tests with coverage report
test-cov:
	uv run pytest tests/ -v --cov=sentimentizer --cov-report=term-missing

## Run only Ray Train tests
test-ray:
	uv run pytest tests/ -v -k "Ray"

## Lint with ruff
lint:
	uv run ruff check .

## Format with black and isort
format:
	uv run black .
	uv run isort .

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

## Build the Docker image
docker-build:
	docker build -t sentimentizer .

## Run the Docker container
docker-run:
	docker run -p 8000:8000 -p 8265:8265 sentimentizer

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

## Remove generated data files, checkpoints, and Python caches
clean:
	rm -rf sentimentizer/data/review_data.parquet
	rm -rf sentimentizer/data/review_data_raw.parquet
	rm -rf sentimentizer/data/weights.pth
	rm -rf checkpoints/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true