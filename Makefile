.PHONY: setup setup-dev download-data train train-rnn train-encoder train-decoder \
       train-distributed train-quick serve test lint format check clean docker-build docker-run \
	   gpu-reset tune tune-rnn tune-encoder tune-decoder tune-standalone \
	   start-metrics stop-metrics setup-dashboards start-exporter stop-exporter stop-ray \
	   upload-rnn upload-encoder upload-decoder download-rnn download-encoder download-decoder \
	   push-hub pull-hub diagnose diagnose-env diagnose-pipeline

# Default device: use auto-detect (cuda > mps > cpu)
DEVICE ?= auto
# Default model type
MODEL ?= rnn
# Default number of lines to load
STOP ?= 300000
# Default run type (new or update)
RUN_TYPE ?= new
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
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) --save

## Train RNN model
train-rnn:
	uv run sentimentizer --model rnn --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) --save

## Train Transformer Encoder model (recommended)
train-encoder:
	uv run sentimentizer --model encoder --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) --save

## Train Transformer Decoder model
train-decoder:
	uv run sentimentizer --model decoder --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) --save

## Quick training run with fewer rows for iteration
train-quick:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop 5000 --save

## Train with checkpointing enabled (saves to CHECKPOINT_DIR, defaults to checkpoints/)
train-checkpoint:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir $(or $(CHECKPOINT_DIR),checkpoints/) --checkpoint-every 1 --save

## Resume training from the latest checkpoint
train-resume:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--checkpoint-dir $(or $(CHECKPOINT_DIR),checkpoints/) --resume-train --save

## Distributed training with Ray Train (2 workers by default)
train-distributed:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--distributed --save

## Distributed training with custom worker count (usage: make train-dist-workers WORKERS=4)
train-dist-workers:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) run --stop $(STOP) \
		--distributed --num-workers $(WORKERS) --save

# ──────────────────────────────────────────────
# Individual pipeline stages
# ──────────────────────────────────────────────

## Extract raw reviews into parquet
extract:
	uv run sentimentizer --model $(MODEL) --run-type $(RUN_TYPE) extract --stop $(STOP)

## Tokenize: build/update dictionary and write processed parquet
tokenize:
	uv run sentimentizer --model $(MODEL) --run-type $(RUN_TYPE) tokenize

## Train only (no extract/tokenize)
train-only:
	uv run sentimentizer --model $(MODEL) --device $(DEVICE) --run-type $(RUN_TYPE) train --save

# ──────────────────────────────────────────────
# Tuning
# ──────────────────────────────────────────────

## Run tuning skill with agent-guided loop and model validation
tune:
	uv run sentimentizer --model $(MODEL) tune --save

## Run tuning skill for RNN
tune-rnn:
	uv run sentimentizer --model rnn tune --save

## Run tuning skill for Encoder
tune-encoder:
	uv run sentimentizer --model encoder tune --save

## Run tuning skill for Decoder
tune-decoder:
	uv run sentimentizer --model decoder tune --save

## Run tuning skill in standalone mode (no LLM agent, single Ray Tune sweep)
tune-standalone:
	uv run sentimentizer --model $(MODEL) tune --mode standalone --save

## Quick tuning test with tiny dataset and few trials
## Usage: make tune-test MODEL=rnn STOP=100 SAMPLES=2
tune-test:
	uv run sentimentizer --model $(MODEL) tune --mode standalone --samples $(SAMPLES) --no-validate --save

## Run tuning skill with custom samples and iterations (usage: make tune-custom SAMPLES=50 ITERATIONS=10)
tune-custom:
	uv run sentimentizer --model $(MODEL) tune --samples $(SAMPLES) --max-iterations $(ITERATIONS) --save

## Run tuning skill without model validation
tune-no-validate:
	uv run sentimentizer --model $(MODEL) tune --no-validate --save

# ──────────────────────────────────────────────
# Hugging Face Hub (push/pull per model)
# ──────────────────────────────────────────────

## Upload RNN weights + dictionary + model card to Hugging Face Hub
upload-rnn:
	uv run sentimentizer --model rnn hf push

## Upload Encoder weights + dictionary + model card to Hugging Face Hub
upload-encoder:
	uv run sentimentizer --model encoder hf push

## Upload Decoder weights + dictionary + model card to Hugging Face Hub
upload-decoder:
	uv run sentimentizer --model decoder hf push

## Upload all models to Hugging Face Hub
push-hub: upload-rnn upload-encoder upload-decoder

## Download RNN weights + dictionary from Hugging Face Hub
download-rnn:
	uv run sentimentizer --model rnn hf pull

## Download Encoder weights + dictionary from Hugging Face Hub
download-encoder:
	uv run sentimentizer --model encoder hf pull

## Download Decoder weights + dictionary from Hugging Face Hub
download-decoder:
	uv run sentimentizer --model decoder hf pull

## Download all models from Hugging Face Hub
pull-hub: download-rnn download-encoder download-decoder

# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

## Fast environment check (no torch/ray imports)
diagnose-env:
	uv run sentimentizer diagnose env

## Full pipeline diagnostics (imports ML stack)
diagnose-pipeline:
	uv run sentimentizer --model $(MODEL) diagnose pipeline

## Run diagnostics (defaults to pipeline)
diagnose:
	uv run sentimentizer --model $(MODEL) diagnose pipeline

# ──────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────

## Start Ray Serve with all three models
serve:
	uv run serve run sentimentizer.serve:app --host 0.0.0.0 --port 8000

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

## Auto-format, auto-fix, then lint (run after every change)
check:
	uv run black .
	uv run ruff check . --fix
	uv run ruff check .

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
# Metrics
# ──────────────────────────────────────────────

## Setup Ray Grafana dashboards using Ray's internal factory
setup-dashboards:
	@mkdir -p metrics/grafana/dashboards
	uv run python scripts/generate_ray_dashboards.py
	@echo "Generated Ray dashboards in metrics/grafana/dashboards/"

## Start the Sentimentizer Prometheus metrics exporter (system, GPU, Ray health)
start-exporter:
	uv run python sentimentizer/exporter.py &

## Stop the Sentimentizer metrics exporter
stop-exporter:
	@pkill -f "sentimentizer/exporter.py" 2>/dev/null || true

## Start Prometheus, Grafana, and metrics exporter for dashboard metrics
start-metrics: setup-dashboards
	cd metrics && docker compose up -d
	@echo "Starting metrics exporter (port 8081)..."
	uv run python sentimentizer/exporter.py &
	@sleep 2
	@echo "All metrics services running. Grafana: http://localhost:3000 (admin/admin)"

## Stop Prometheus, Grafana, and metrics exporter
stop-metrics:
	@pkill -f "sentimentizer/exporter.py" 2>/dev/null || true
	cd metrics && docker compose down

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

## Force stop all local Ray instances
stop-ray:
	uv run ray stop --force

## Remove generated data files, checkpoints, and Python caches
clean: stop-ray
	rm -rf sentimentizer/data/review_data.parquet
	rm -rf sentimentizer/data/review_data_raw.parquet
	rm -rf sentimentizer/data/weights.pth
	rm -rf checkpoints/
	rm -rf tuning_results/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Cleaning Ray temporary files..."
	rm -rf /tmp/ray/*
	@echo "==> Cleaning Ray Tune results..."
	rm -rf ~/ray_results/*

## Clean only Ray-related files and logs
clean-ray: stop-ray
	rm -rf /tmp/ray/*
	rm -rf ~/ray_results/*

# Fix NVIDIA driver/library mismatch without rebooting
gpu-reset:
	@echo "==> Stopping services that might use the GPU..."
	-sudo systemctl stop ollama
	-sudo systemctl stop docker
	@echo "==> Unloading NVIDIA kernel modules..."
	-sudo rmmod nvidia_drm
	-sudo rmmod nvidia_modeset
	-sudo rmmod nvidia_uvm
	-sudo rmmod nvidia
	@echo "==> Reloading NVIDIA kernel modules..."
	sudo modprobe nvidia
	sudo modprobe nvidia_uvm
	@echo "==> Restarting services..."
	-sudo systemctl start docker
	-sudo systemctl start ollama
	@echo "==> Verifying NVML initialization..."
	nvidia-smi