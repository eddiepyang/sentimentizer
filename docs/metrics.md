# Metrics Pipeline

Sentimentizer exposes training and system metrics via Prometheus for visualization in Grafana. This document describes the architecture, configuration, and troubleshooting of the metrics pipeline.

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│  Training Driver         │     │  Standalone Exporter      │     │  Ray Workers  │
│  (port 8083)             │     │  (port 8081)              │     │  (port 8080)  │
│                          │     │                            │     │               │
│  sentimentizer_training_*│     │  sentimentizer_training_* │     │  ray_sentimen-│
│  gauges (set during      │     │  gauges (initialized to 0, │     │  tizer_live_* │
│  training + from JSON)   │     │  updated from JSON file)   │     │  gauges       │
│                          │     │  sentimentizer_system_*    │     │               │
│  /tmp/sentimentizer_     │────▶│  sentimentizer_gpu_*      │     │               │
│  training_metrics.json   │     │  sentimentizer_ray_*      │     │               │
└──────────┬───────────────┘     └──────────┬───────────────┘     └──────┬────────┘
           │                                │                             │
           │    ┌──────────────────────┐    │                             │
           └───▶│  Prometheus           │◀──┘                             │
                │  (port 9090)         │◀────────────────────────────────┘
                │  Scrape targets:      │
                │  - sentimentizer:8081 │
                │  - sentimentizer-      │
                │    training:8083      │
                │  - ray:8080           │
                │  - sentimentizer-      │
                │    tune:8082           │
                └────────┬─────────────┘
                         │
                         ▼
                ┌──────────────────────┐
                │  Grafana              │
                │  (port 3000)          │
                │  Dashboard:           │
                │  sentimentizerTraining│
                └──────────────────────┘
```

## Prometheus Scrape Targets

| Job Name | Port | Purpose |
|----------|------|---------|
| `sentimentizer` | 8081 | Always-on exporter: system metrics, GPU stats, Ray health, and persistent training gauges |
| `sentimentizer-training` | 8083 | Driver process: live training gauges during `make train` or `make train-distributed` |
| `sentimentizer-tune` | 8082 | Tuning process: Ray Tune trial metrics |
| `ray` | 8080 | Ray cluster metrics including `ray_sentimentizer_live_*` gauges from distributed workers |

## Metric Naming Convention

### `sentimentizer_training_*` (final metrics)
Set by the driver process and standalone exporter. These persist after training completes.

- `sentimentizer_training_train_loss{model_type}`
- `sentimentizer_training_val_loss{model_type}`
- `sentimentizer_training_val_accuracy{model_type}`
- `sentimentizer_training_val_precision{model_type}`
- `sentimentizer_training_val_recall{model_type}`
- `sentimentizer_training_val_f1{model_type}`
- `sentimentizer_training_val_cohen_kappa{model_type}`
- `sentimentizer_training_val_auc_roc{model_type}`
- `sentimentizer_training_val_positive_accuracy{model_type}`
- `sentimentizer_training_val_negative_accuracy{model_type}`
- `sentimentizer_training_epoch{model_type}`

The `model_type` label is one of: `rnn`, `encoder`, `decoder`.

### `ray_sentimentizer_live_*` (real-time distributed metrics)
Set by Ray workers during distributed training. These are prefixed with `ray_` by Ray's metrics system and only exist while Ray is running.

- `ray_sentimentizer_live_train_loss{model_type}`
- `ray_sentimentizer_live_val_loss{model_type}`
- `ray_sentimentizer_live_val_accuracy{model_type}`
- `ray_sentimentizer_live_val_precision{model_type}`
- `ray_sentimentizer_live_val_recall{model_type}`
- `ray_sentimentizer_live_val_f1{model_type}`
- `ray_sentimentizer_live_val_cohen_kappa{model_type}`
- `ray_sentimentizer_live_val_auc_roc{model_type}`
- `ray_sentimentizer_live_val_positive_accuracy{model_type}`
- `ray_sentimentizer_live_val_negative_accuracy{model_type}`

### System metrics (`sentimentizer_system_*`)
Always available from port 8081:

- `sentimentizer_system_cpu_percent`
- `sentimentizer_system_memory_percent`
- `sentimentizer_system_memory_available_bytes`
- `sentimentizer_system_memory_total_bytes`
- `sentimentizer_system_disk_percent`
- `sentimentizer_system_disk_free_bytes`
- `sentimentizer_system_disk_total_bytes`
- `sentimentizer_gpu_utilization_percent{gpu_index, gpu_name}`
- `sentimentizer_gpu_memory_used_bytes{gpu_index, gpu_name}`
- `sentimentizer_gpu_memory_total_bytes{gpu_index, gpu_name}`
- `sentimentizer_gpu_temperature_celsius{gpu_index, gpu_name}`
- `sentimentizer_ray_available`
- `sentimentizer_ray_node_count`
- `sentimentizer_ray_metric_count`

## How Training Metrics Flow

### During Distributed Training (`make train-distributed`)

1. **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges every 100 training steps from rank 0
2. **Driver process** (port 8083) sets `sentimentizer_training_*` gauges at the end of each epoch
3. **Prometheus** scrapes both ports every 10 seconds
4. **Grafana** queries `sentimentizer_training_* or ray_sentimentizer_live_*` to show whichever data is available

### After Training Completes

1. **Driver process** writes final metrics to `/tmp/sentimentizer_training_metrics.json`
2. **Driver process** exits (port 8083 goes offline)
3. **Standalone exporter** (port 8081, always-on) reads the JSON file every 10 seconds and updates its own `sentimentizer_training_*` gauges
4. **Prometheus** continues scraping port 8081, picking up the persisted metrics
5. **Grafana** shows the final training results even after training is done

### During Single-Node Training (`make train`)

1. **Trainer** (in the driver process) sets `sentimentizer_training_*` gauges at each validation epoch
2. **Driver process** serves them on port 8083
3. Same persistence flow as above when training completes

## NaN Handling in Metrics

During training, `torch.sigmoid()` on extreme logit values can produce NaN, which causes `sklearn.metrics.roc_auc_score()` to raise `ValueError: Input contains NaN`. The codebase handles this at multiple levels:

1. **`sentimentizer/metrics.py`**: `compute_classification_metrics()` and `_auc_roc()` replace NaN probabilities with 0.5 (random-guess probability) before calling sklearn
2. **`sentimentizer/metrics.py`**: `compute_metrics_from_model()` replaces NaN at the tensor level using `torch.where()` before converting to numpy
3. **`sentimentizer/trainer.py`**: Both `Trainer.evaluate()` and `_train_func()` (Ray) replace NaN in probabilities with 0.5 after `torch.sigmoid()`

## Starting the Metrics Stack

```bash
# Start Prometheus + Grafana
cd metrics && docker compose up -d

# Start the standalone metrics exporter (port 8081)
make start-exporter
# or: uv run python sentimentizer/exporter.py

# Start training (driver starts port 8083 automatically)
make train-distributed
```

## Grafana Dashboard

The `sentimentizerTraining` dashboard (UID: `sentimentizerTraining`) shows:

- **Loss**: Training + validation loss over time
- **Accuracy**: Overall, positive-class, and negative-class accuracy
- **Precision / Recall / F1**: Classification metrics
- **Cohen's Kappa / AUC-ROC**: Agreement and discrimination metrics
- **Training Epoch**: Current epoch number

The dashboard uses PromQL `or` expressions to fall back between data sources:
```promql
# Shows training metrics if available, falls back to live Ray metrics
sentimentizer_training_val_accuracy{model_type=~"$model_type"}
  or ray_sentimentizer_live_val_accuracy{model_type=~"$model_type"}
```

## Troubleshooting

### Dashboard shows "No data"

1. **Check Prometheus targets**: Visit `http://localhost:9090/targets` — all targets should be UP (except `sentimentizer-training` and `sentimentizer-tune` which are only active during training)
2. **Check exporter is running**: `curl http://localhost:8081/metrics | grep sentimentizer_training` should show initialized gauges
3. **Check training completed**: `cat /tmp/sentimentizer_training_metrics.json` should contain the final metrics
4. **Check dashboard queries**: The dashboard uses `ray_` prefix for Ray metrics (`ray_sentimentizer_live_*`), not `sentimentizer_live_*`

### Metrics disappear after training ends

This is expected — the driver process (port 8083) exits. The standalone exporter (port 8081) reads persisted metrics from `/tmp/sentimentizer_training_metrics.json` every 10 seconds. If the file doesn't exist, gauges show 0.

### "ValueError: Input contains NaN" crash

This has been fixed. NaN values in probabilities (from extreme logit values) are now replaced with 0.5 at multiple levels. See the "NaN Handling" section above.

### Ray metrics not appearing

Ray metrics are only available while Ray is running. Check that `ray` is in the UP state at `http://localhost:9090/targets`. The `ray_sentimentizer_live_*` gauges are set by rank 0 workers every 100 training steps.

### Port conflicts

- Port 8080: Ray metrics (auto-configured by `ray.init(_metrics_export_port=8080)`)
- Port 8081: Standalone exporter (`sentimentizer/exporter.py`)
- Port 8082: Tune metrics (started by `tune_model()`)
- Port 8083: Driver training metrics (started by `_ensure_metrics_server()`)

If a port is already in use, the code logs a warning and continues — `prometheus_client` gauges are process-global, so metrics still work within the same process.