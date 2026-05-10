# Metrics Pipeline

Sentimentizer exposes training and system metrics via Prometheus for visualization in Grafana. This document describes the architecture, configuration, and troubleshooting of the metrics pipeline.

## Known Gaps

The following gaps were identified between this document and the current codebase. All have been resolved:

1. ~~**Gauge pre-initialization claim was incorrect**~~ — Fixed: updated to reflect that the standalone exporter creates training Gauge objects at module-import time (they appear with value `0` immediately).

2. ~~**Missing `ray_sentimentizer_live_epoch` metric**~~ — Fixed: added to the Ray live metrics list.

3. ~~**Missing Ray controller health metrics**~~ — Fixed: added `sentimentizer_ray_controller_state` and `sentimentizer_ray_controller_operation_time_s` to the system metrics section.

4. ~~**Tune metrics entirely undocumented**~~ — Fixed: added a full `sentimentizer_tune_*` section documenting all 15 tuning metrics (10 per-trial + 5 aggregate).

5. ~~**Missing `sentimentizer_system_info` metric**~~ — Fixed: added to the system metrics section as an Info-type metric.

6. ~~**Minor NaN handling imprecision**~~ — Fixed: updated to mention that `_auc_roc()` falls back to a manual trapezoidal implementation when sklearn is unavailable.

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐
│  Training Driver      │     │  Standalone Exporter  │
│                       │     │  (port 8081)           │
│  Writes metrics to    │────▶│  Reads JSON file every │
│  /tmp/sentimentizer_  │     │  10s, serves gauges    │
│  training_metrics.json│     │  + system/GPU/Ray stats │
└───────────────────────┘     └──────────┬───────────┘
                                         │
┌──────────────────────┐                  │
│  Ray Workers         │                  │
│  (port 8080)         │                  │
│                      │                  │
│  ray_sentimentizer_  │                  │
│  live_* gauges       │                  │
└──────────┬───────────┘                  │
           │                              │
           │    ┌──────────────────────┐   │
           └───▶│  Prometheus           │◀──┘
                │  (port 9090)         │
                │  Scrape targets:      │
                │  - sentimentizer:8081 │
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
| `sentimentizer` | 8081 | Always-on exporter: system metrics, GPU stats, Ray health, and training gauges (loaded from JSON) |
| `sentimentizer-tune` | 8082 | Tuning process: Ray Tune trial metrics (active during tuning only) |
| `ray` | 8080 | Ray cluster metrics including `ray_sentimentizer_live_*` gauges from distributed workers |

## Metric Naming Convention

### `sentimentizer_training_*` (training metrics)
Gauges are pre-initialized at module-import time in the standalone exporter (`exporter.py`) and appear in Prometheus with value `0` as soon as the exporter starts. Values are updated at each validation epoch and persisted to the JSON file when training completes.

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
- `ray_sentimentizer_live_epoch{model_type}`

### `sentimentizer_tune_*` (tuning metrics)
Emitted on port 8082 during Ray Tune hyperparameter tuning runs. These metrics are only available while `tune_model()` is actively running.

#### Per-trial metrics (labeled by `trial_id` and `model_type`):
- `sentimentizer_tune_val_accuracy{trial_id, model_type}`
- `sentimentizer_tune_val_loss{trial_id, model_type}`
- `sentimentizer_tune_train_loss{trial_id, model_type}`
- `sentimentizer_tune_val_f1{trial_id, model_type}`
- `sentimentizer_tune_val_cohen_kappa{trial_id, model_type}`
- `sentimentizer_tune_val_precision{trial_id, model_type}`
- `sentimentizer_tune_val_recall{trial_id, model_type}`
- `sentimentizer_tune_val_positive_accuracy{trial_id, model_type}`
- `sentimentizer_tune_val_negative_accuracy{trial_id, model_type}`
- `sentimentizer_tune_epoch{trial_id, model_type}`

#### Aggregate metrics (labeled by `model_type` only):
- `sentimentizer_tune_best_val_accuracy{model_type}`
- `sentimentizer_tune_best_val_loss{model_type}`
- `sentimentizer_tune_best_val_f1{model_type}`
- `sentimentizer_tune_trial_count{model_type}`
- `sentimentizer_tune_trial_completed_count{model_type}`

### System metrics (`sentimentizer_system_*`)
Always available from port 8081:

- `sentimentizer_system_info{platform, python, cpu_count}` — Info-type metric with static system metadata
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
- `sentimentizer_ray_controller_state`
- `sentimentizer_ray_controller_operation_time_s`

## How Training Metrics Flow

### During Distributed Training (`make train-distributed`)

1. **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges every 100 training steps from rank 0
2. **Driver process** sets `sentimentizer_training_*` gauges in-process and writes final metrics to `/tmp/sentimentizer_training_metrics.json`
3. **Standalone exporter** (port 8081) reads the JSON file every 10 seconds and updates its gauges
4. **Prometheus** scrapes ports 8080 and 8081 every 10 seconds
5. **Grafana** queries `sentimentizer_training_* or ray_sentimentizer_live_*` to show whichever data is available

### After Training Completes

1. **Driver process** writes final metrics to `/tmp/sentimentizer_training_metrics.json`
2. **Driver process** exits
3. **Standalone exporter** (port 8081, always-on) reads the JSON file every 10 seconds and updates its `sentimentizer_training_*` gauges
4. **Prometheus** continues scraping port 8081, picking up the persisted metrics
5. **Grafana** shows the final training results even after training is done

### During Single-Node Training (`make train`)

1. **Trainer** (in the driver process) sets `sentimentizer_training_*` gauges at each validation epoch and writes metrics to JSON
2. **Standalone exporter** (port 8081) reads the JSON file and serves the gauges
3. Same persistence flow as above when training completes

## NaN Handling in Metrics

During training, `torch.sigmoid()` on extreme logit values can produce NaN, which causes `sklearn.metrics.roc_auc_score()` to raise `ValueError: Input contains NaN`. The codebase handles this at multiple levels:

1. **`sentimentizer/metrics.py`**: `compute_classification_metrics()` and `_auc_roc()` replace NaN probabilities with 0.5 (random-guess probability) before computing AUC-ROC (uses `sklearn.metrics.roc_auc_score` when available, falls back to a manual trapezoidal implementation otherwise)
2. **`sentimentizer/metrics.py`**: `compute_metrics_from_model()` replaces NaN at the tensor level using `torch.where()` before converting to numpy
3. **`sentimentizer/trainer.py`**: Both `Trainer.evaluate()` and `_train_func()` (Ray) replace NaN in probabilities with 0.5 after `torch.sigmoid()`

## Starting the Metrics Stack

```bash
# Start Prometheus + Grafana
cd metrics && docker compose up -d

# Start the standalone metrics exporter (port 8081)
make start-exporter
# or: uv run python sentimentizer/exporter.py

# Start training (driver writes metrics to JSON file automatically)
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

1. **Check Prometheus targets**: Visit `http://localhost:9090/targets` — `sentimentizer` and `ray` should be UP
2. **Check exporter is running**: `curl http://localhost:8081/metrics | grep sentimentizer_training` should show metrics for models that have completed training
3. **Check training completed**: `cat /tmp/sentimentizer_training_metrics.json` should contain the final metrics
4. **Check dashboard queries**: The dashboard uses `ray_` prefix for Ray metrics (`ray_sentimentizer_live_*`), not `sentimentizer_live_*`

### Metrics disappear after training ends

The standalone exporter (port 8081) reads persisted metrics from `/tmp/sentimentizer_training_metrics.json` every 10 seconds. If the file doesn't exist or is empty, training gauges will not appear.

### "ValueError: Input contains NaN" crash

This has been fixed. NaN values in probabilities (from extreme logit values) are now replaced with 0.5 at multiple levels. See the "NaN Handling" section above.

### Ray metrics not appearing

Ray metrics are only available while Ray is running. Check that `ray` is in the UP state at `http://localhost:9090/targets`. The `ray_sentimentizer_live_*` gauges are set by rank 0 workers every 100 training steps.

### Port conflicts

- Port 8080: Ray metrics (auto-configured by `ray.init(_metrics_export_port=8080)`)
- Port 8081: Standalone exporter (`sentimentizer/exporter.py`)
- Port 8082: Tune metrics (started by `tune_model()`)

If a port is already in use, the code logs a warning and continues — `prometheus_client` gauges are process-global, so metrics still work within the same process.

### Grafana PromQL Parse Errors (e.g., `unexpected ","`)

If you encounter syntax or parse errors in Grafana after modifying the dashboard generator scripts, check the following common pitfalls:

1. **Invalid `label_values()` macro usage**: Grafana's Prometheus datasource plugin does **not** support combining multiple `label_values()` macro calls using `+` or `OR` in template variable queries (e.g., `label_values(metric_A, label) + label_values(metric_B, label)`). This will crash Grafana's PromQL parser. The correct format is to use a single `label_values()` call referencing a highly reliable metric (such as `ray_node_cpu_count` which is emitted universally) instead of trying to chain fragile metrics.
2. **Hanging commas from stripped filters**: Ray's internal dashboard generators use Jinja2-style placeholders (like `{{{global_filters}}}`). When scripts strip these placeholders out during patching, it can leave behind leading commas (`{, label="x"}`) or consecutive commas (`{x="1", , y="2"}`). The `scripts/generate_ray_dashboards.py` includes robust regex sanitization to clean these artifacts natively.