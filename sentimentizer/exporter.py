"""Standalone Prometheus metrics exporter for Sentimentizer.

Exposes system, GPU, and application metrics on a configurable port
for Prometheus scraping. This ensures the Grafana dashboard always has
data to display, even when Ray is not running.

Usage:
    uv run python sentimentizer/exporter.py [--port 8081] [--interval 10]
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import threading
import time

import psutil
from prometheus_client import Gauge, Info, start_http_server

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System metrics
# ──────────────────────────────────────────────

SYSTEM_CPU_PERCENT = Gauge(
    "sentimentizer_system_cpu_percent",
    "System CPU usage percentage",
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "sentimentizer_system_memory_percent",
    "System memory usage percentage",
)

SYSTEM_MEMORY_AVAILABLE_BYTES = Gauge(
    "sentimentizer_system_memory_available_bytes",
    "Available system memory in bytes",
)

SYSTEM_MEMORY_TOTAL_BYTES = Gauge(
    "sentimentizer_system_memory_total_bytes",
    "Total system memory in bytes",
)

SYSTEM_DISK_PERCENT = Gauge(
    "sentimentizer_system_disk_percent",
    "System disk usage percentage for root partition",
)

SYSTEM_DISK_FREE_BYTES = Gauge(
    "sentimentizer_system_disk_free_bytes",
    "Free disk space in bytes for root partition",
)

SYSTEM_DISK_TOTAL_BYTES = Gauge(
    "sentimentizer_system_disk_total_bytes",
    "Total disk space in bytes for root partition",
)

SYSTEM_INFO = Info(
    "sentimentizer_system",
    "System information",
)

# ──────────────────────────────────────────────
# GPU metrics
# ──────────────────────────────────────────────

GPU_UTILIZATION = Gauge(
    "sentimentizer_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_index", "gpu_name"],
)

GPU_MEMORY_USED_BYTES = Gauge(
    "sentimentizer_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_index", "gpu_name"],
)

GPU_MEMORY_TOTAL_BYTES = Gauge(
    "sentimentizer_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    ["gpu_index", "gpu_name"],
)

GPU_TEMPERATURE_CELSIUS = Gauge(
    "sentimentizer_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_index", "gpu_name"],
)

# ──────────────────────────────────────────────
# Ray Tune trial metrics
# ──────────────────────────────────────────────

TUNE_TRIAL_VAL_ACCURACY = Gauge(
    "sentimentizer_tune_val_accuracy",
    "Validation accuracy reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_LOSS = Gauge(
    "sentimentizer_tune_val_loss",
    "Validation loss reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_TRAIN_LOSS = Gauge(
    "sentimentizer_tune_train_loss",
    "Training loss reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_F1 = Gauge(
    "sentimentizer_tune_val_f1",
    "Validation F1 score reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_COHEN_KAPPA = Gauge(
    "sentimentizer_tune_val_cohen_kappa",
    "Validation Cohen's kappa reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_PRECISION = Gauge(
    "sentimentizer_tune_val_precision",
    "Validation precision reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_VAL_RECALL = Gauge(
    "sentimentizer_tune_val_recall",
    "Validation recall reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_POSITIVE_ACCURACY = Gauge(
    "sentimentizer_tune_val_positive_accuracy",
    "Validation positive-class accuracy reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_NEGATIVE_ACCURACY = Gauge(
    "sentimentizer_tune_val_negative_accuracy",
    "Validation negative-class accuracy reported by the current Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_TRIAL_EPOCH = Gauge(
    "sentimentizer_tune_epoch",
    "Current training epoch for the Ray Tune trial",
    ["trial_id", "model_type"],
)

TUNE_BEST_VAL_ACCURACY = Gauge(
    "sentimentizer_tune_best_val_accuracy",
    "Best validation accuracy across all Ray Tune trials",
    ["model_type"],
)

TUNE_BEST_VAL_LOSS = Gauge(
    "sentimentizer_tune_best_val_loss",
    "Best validation loss across all Ray Tune trials",
    ["model_type"],
)

TUNE_BEST_VAL_F1 = Gauge(
    "sentimentizer_tune_best_val_f1",
    "Best validation F1 score across all Ray Tune trials",
    ["model_type"],
)

TUNE_TRIAL_COUNT = Gauge(
    "sentimentizer_tune_trial_count",
    "Total number of Ray Tune trials",
    ["model_type"],
)

TUNE_TRIAL_COMPLETED_COUNT = Gauge(
    "sentimentizer_tune_trial_completed_count",
    "Number of completed Ray Tune trials",
    ["model_type"],
)

# ──────────────────────────────────────────────
# Training model performance metrics
# ──────────────────────────────────────────────

TRAINING_TRAIN_LOSS = Gauge(
    "sentimentizer_training_train_loss",
    "Training loss from the current training run",
    ["model_type"],
)

TRAINING_VAL_LOSS = Gauge(
    "sentimentizer_training_val_loss",
    "Validation loss from the current training run",
    ["model_type"],
)

TRAINING_VAL_ACCURACY = Gauge(
    "sentimentizer_training_val_accuracy",
    "Validation accuracy from the current training run",
    ["model_type"],
)

TRAINING_VAL_PRECISION = Gauge(
    "sentimentizer_training_val_precision",
    "Validation precision (positive class) from the current training run",
    ["model_type"],
)

TRAINING_VAL_RECALL = Gauge(
    "sentimentizer_training_val_recall",
    "Validation recall (positive class) from the current training run",
    ["model_type"],
)

TRAINING_VAL_F1 = Gauge(
    "sentimentizer_training_val_f1",
    "Validation F1 score from the current training run",
    ["model_type"],
)

TRAINING_VAL_COHEN_KAPPA = Gauge(
    "sentimentizer_training_val_cohen_kappa",
    "Validation Cohen's kappa from the current training run",
    ["model_type"],
)

TRAINING_VAL_AUC_ROC = Gauge(
    "sentimentizer_training_val_auc_roc",
    "Validation AUC-ROC from the current training run",
    ["model_type"],
)

TRAINING_VAL_POSITIVE_ACCURACY = Gauge(
    "sentimentizer_training_val_positive_accuracy",
    "Validation positive-class accuracy from the current training run",
    ["model_type"],
)

TRAINING_VAL_NEGATIVE_ACCURACY = Gauge(
    "sentimentizer_training_val_negative_accuracy",
    "Validation negative-class accuracy from the current training run",
    ["model_type"],
)

TRAINING_EPOCH = Gauge(
    "sentimentizer_training_epoch",
    "Current training epoch number",
    ["model_type"],
)

# ──────────────────────────────────────────────
# Ray health metrics
# ──────────────────────────────────────────────

RAY_AVAILABLE = Gauge(
    "sentimentizer_ray_available",
    "Whether Ray cluster is reachable (1=up, 0=down)",
)

RAY_NODE_COUNT = Gauge(
    "sentimentizer_ray_node_count",
    "Number of Ray nodes detected from metrics",
)

RAY_METRIC_COUNT = Gauge(
    "sentimentizer_ray_metric_count",
    "Number of Ray metric series found",
)

RAY_CONTROLLER_STATE = Gauge(
    "sentimentizer_ray_controller_state",
    "Ray controller state",
)

RAY_CONTROLLER_OPERATION_TIME = Gauge(
    "sentimentizer_ray_controller_operation_time_s",
    "Ray controller operation time",
)


def _update_system_metrics() -> None:
    """Update system metric gauges from psutil."""
    SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=0))

    mem = psutil.virtual_memory()
    SYSTEM_MEMORY_PERCENT.set(mem.percent)
    SYSTEM_MEMORY_AVAILABLE_BYTES.set(mem.available)
    SYSTEM_MEMORY_TOTAL_BYTES.set(mem.total)

    disk = psutil.disk_usage("/")
    SYSTEM_DISK_PERCENT.set(disk.percent)
    SYSTEM_DISK_FREE_BYTES.set(disk.free)
    SYSTEM_DISK_TOTAL_BYTES.set(disk.total)


def _update_gpu_metrics() -> None:
    """Update GPU metric gauges from nvidia-smi if available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            gpu_index = parts[0]
            gpu_name = parts[1]
            try:
                utilization = float(parts[2]) if parts[2] != "[N/A]" else 0
                mem_used_mib = float(parts[3]) if parts[3] != "[N/A]" else 0
                mem_total_mib = float(parts[4]) if parts[4] != "[N/A]" else 0
                temperature = float(parts[5]) if parts[5] != "[N/A]" else 0
            except ValueError:
                continue

            GPU_UTILIZATION.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(utilization)
            GPU_MEMORY_USED_BYTES.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(
                mem_used_mib * 1024 * 1024
            )
            GPU_MEMORY_TOTAL_BYTES.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(
                mem_total_mib * 1024 * 1024
            )
            GPU_TEMPERATURE_CELSIUS.labels(gpu_index=gpu_index, gpu_name=gpu_name).set(temperature)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _update_ray_metrics(ray_url: str) -> None:
    """Check Ray metrics endpoint and update Ray health gauges."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{ray_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
            metric_lines = [
                line for line in text.split("\n") if line.strip() and not line.startswith("#")
            ]
            RAY_AVAILABLE.set(1)
            RAY_METRIC_COUNT.set(len(metric_lines))

            # Count unique nodes by looking for ray_node_cpu_percent
            node_instances = set()
            for line in metric_lines:
                if line.startswith("ray_node_cpu_percent") and 'instance="' in line:
                    start = line.index('instance="') + len('instance="')
                    end = line.index('"', start)
                    node_instances.add(line[start:end])
                elif line.startswith(
                    (
                        "ray_serve_controller_state",
                        "ray_controller_state",
                    )
                ):
                    with contextlib.suppress(ValueError):
                        RAY_CONTROLLER_STATE.set(float(line.split()[-1]))
                elif line.startswith(
                    (
                        "ray_serve_controller_operation_time",
                        "ray_controller_operation_time",
                    )
                ):
                    with contextlib.suppress(ValueError):
                        RAY_CONTROLLER_OPERATION_TIME.set(float(line.split()[-1]))
            RAY_NODE_COUNT.set(len(node_instances))
    except (urllib.error.URLError, OSError):
        RAY_AVAILABLE.set(0)
        RAY_NODE_COUNT.set(0)
        RAY_METRIC_COUNT.set(0)
        RAY_CONTROLLER_STATE.set(0)
        RAY_CONTROLLER_OPERATION_TIME.set(0)


def _metrics_loop(interval: int, ray_url: str) -> None:
    """Background thread that periodically updates metric gauges."""
    while True:
        try:
            _update_system_metrics()
        except Exception as e:
            logger.warning("Error updating system metrics: %s", e)
        try:
            _update_gpu_metrics()
        except Exception as e:
            logger.warning("Error updating GPU metrics: %s", e)
        try:
            _update_ray_metrics(ray_url)
        except Exception as e:
            logger.warning("Error updating Ray metrics: %s", e)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentimentizer Prometheus metrics exporter")
    parser.add_argument(
        "--port", type=int, default=8081, help="Port to serve metrics on (default: 8081)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Metrics collection interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--ray-url",
        default="http://localhost:8080",
        help="Ray metrics URL (default: http://localhost:8080)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Set static system info
    SYSTEM_INFO.info(
        {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cpu_count": str(os.cpu_count() or 0),
        }
    )

    # Initial collection before starting server
    _update_system_metrics()
    _update_gpu_metrics()
    _update_ray_metrics(args.ray_url)

    # Start background collection thread
    collector = threading.Thread(
        target=_metrics_loop,
        args=(args.interval, args.ray_url),
        daemon=True,
    )
    collector.start()

    # Start Prometheus HTTP server
    start_http_server(args.port)
    logger.info("Sentimentizer metrics exporter started on port %d", args.port)
    logger.info("Collecting metrics every %d seconds", args.interval)
    logger.info("Ray metrics URL: %s", args.ray_url)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter")


if __name__ == "__main__":
    main()
