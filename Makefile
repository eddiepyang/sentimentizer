.PHONY: setup setup-dev download-data train train-rnn train-encoder train-decoder \
       train-distributed train-quick serve test lint format check clean docker-build docker-run \
	   gpu-reset tune tune-rnn tune-encoder tune-decoder tune-standalone \
	   start-metrics stop-metrics setup-dashboards start-exporter stop-exporter stop-ray

# Default device: use auto-detect (cuda > mps > cpu)
DEVICE ?= auto
# Default model type
MODEL ?= rnn
# Default number of lines to load
STOP ?= 300000
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
# Tuning Skill (tune + validate until good model)
# ──────────────────────────────────────────────

## Run tuning skill with agent-guided loop and model validation
tune:
	uv run python workflows/driver.py --model $(MODEL) --tune --save

## Run tuning skill for RNN
tune-rnn:
	uv run python workflows/driver.py --model rnn --tune --save

## Run tuning skill for Encoder
tune-encoder:
	uv run python workflows/driver.py --model encoder --tune --save

## Run tuning skill for Decoder
tune-decoder:
	uv run python workflows/driver.py --model decoder --tune --save

## Run tuning skill in standalone mode (no LLM agent, single Ray Tune sweep)
tune-standalone:
	uv run python workflows/driver.py --model $(MODEL) --tune --tune-mode standalone --save

## Run tuning skill with custom samples and iterations (usage: make tune-custom SAMPLES=50 ITERATIONS=10)
tune-custom:
	uv run python workflows/driver.py --model $(MODEL) --tune --tune-samples $(SAMPLES) --tune-max-iterations $(ITERATIONS) --save

## Run tuning skill without model validation
tune-no-validate:
	uv run python workflows/driver.py --model $(MODEL) --tune --no-validate --save

## Upload RNN weights to Hugging Face Hub
upload-rnn:
	uv run python -c "from sentimentizer.hf import push_model_to_hub; push_model_to_hub('sentimentizer/data/rnn_weights.pth', 'ryeyoo/sentimentizer-rnn', 'rnn')"

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
clean:
	rm -rf sentimentizer/data/review_data.parquet
	rm -rf sentimentizer/data/review_data_raw.parquet
	rm -rf sentimentizer/data/weights.pth
	rm -rf checkpoints/
	rm -rf tuning_results/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

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