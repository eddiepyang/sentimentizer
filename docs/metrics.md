# Metrics Pipeline

Sentimentizer exposes training and system metrics via Prometheus for visualization in Grafana. This document describes the architecture, configuration, and troubleshooting of the metrics pipeline.

## Known Gaps

The following gaps were identified between this document and the current codebase. All have been resolved:

1. ~~**Gauge pre-initialization claim was incorrect**~~ — Fixed: updated to reflect that the standalone exporter creates training Gauge objects at module-import time (they appear with value `0` immediately).

2. ~~**Missing `ray_sentimentizer_live_epoch` metric**~~ — Fixed: added to the Ray live metrics list.

3. ~~**Missing Ray controller health metrics**~~ — Fixed: added `sentimentizer_ray_controller_state` and `sentimentizer_ray_controller_operation_time_s` to the system metrics section.

4. ~~**Tune metrics entirely undocumented**~~ — Fixed: added a full `sentimentizer_tune_*` section documenting all 15 tuning metrics (10 per-trial + 5 aggregate).

5. ~~**Missing `sentimentizer_system_info` metric**~~ — Fixed: added to the system metrics section as an Info-type metric.

6. ~~**Minor NaN handling imprecision**~~ — Fixed: NaN handling now uses `torchmetrics.BinaryAUROC` with explicit NaN→0.5 replacement before computation.

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐
│  Training Driver      │     │  Standalone Exporter  │
│                       │     │  (port 8081)           │
│  Writes per-model    │────▶│  Reads per-model JSON │
│  JSON files to        │     │  files every 10s,     │
│  /tmp/sentimentizer_ │     │  serves gauges        │
│  metrics/*.json       │     │  + system/GPU/Ray     │
│                       │     │  stats                │
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
                 │  Dashboard:            │
                 │  sentimentizerModel-  │
                 │  Training              │
                └──────────────────────┘
```

## Prometheus Scrape Targets

| Job Name | Port | Purpose |
|----------|------|---------|
| `sentimentizer` | 8081 | Always-on exporter: system metrics, GPU stats, Ray health, and training gauges (loaded from JSON) |
| `sentimentizer-tune` | 8082 | Tuning process: Ray Tune trial metrics (active during tuning only) |
| `ray` | 8080 | Ray cluster metrics including `ray_sentimentizer_live_*` gauges from distributed workers |

## Metric Definitions

### Core Classification Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **Accuracy** | 0–1 | Overall correctness: (TP + TN) / total |
| **Precision** | 0–1 | Positive-class precision: TP / (TP + FP). Of all samples predicted positive, how many were actually positive? |
| **Recall** | 0–1 | Positive-class recall (sensitivity): TP / (TP + FN). Of all actual positives, how many were correctly identified? |
| **F1** | 0–1 | Harmonic mean of precision and recall (positive class only). Balances false positives and false negatives for the positive class. |
| **Positive Accuracy** | 0–1 | Accuracy on positive samples only: TP / (TP + FN). Same as recall for the positive class. |
| **Negative Accuracy** | 0–1 | Accuracy on negative samples only: TN / (TN + FP). Same as specificity for the negative class. |

### Agreement and Correlation Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **Cohen's Kappa** | -1–1 | Agreement beyond chance. 0 = random agreement, 1 = perfect agreement. Unlike accuracy, accounts for agreement expected by chance. Returns 0.0 for single-class targets (torchmetrics returns NaN, coerced to 0.0). |
| **MCC** (Matthews Correlation Coefficient) | -1–1 | Best single-number summary of confusion matrix quality. Uses all four cells (TP, TN, FP, FN) symmetrically, making it robust to class imbalance where accuracy can be misleading. 0 = random, 1 = perfect, -1 = inverse. |

### Probability-Based Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **AUC-ROC** | 0–1 | Area under the Receiver Operating Characteristic curve. Measures ranking quality: how well the model separates positive from negative samples across all thresholds. 0.5 = random, 1.0 = perfect separation. Requires probability scores; None when not available. Returns 0.0 for single-class targets. |
| **Average Precision** (PR-AUC) | 0–1 | Area under the Precision-Recall curve. More informative than AUC-ROC for imbalanced datasets because it focuses on the positive class. A random classifier gives AP = positive prevalence (not 0.5), making it a stricter test. Requires probability scores; None when not available. Returns 0.0 for single-class targets. |

### Negative-Class Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **NPV** (Negative Predictive Value) | 0–1 | TN / (TN + FN). Of all samples predicted negative, how many were actually negative? Complement to precision for the negative class. Low NPV indicates many false negatives — the model incorrectly classifies actual negatives as positive. |
| **Macro F1** | 0–1 | Mean of per-class F1 scores (F1_positive + F1_negative) / 2. Weights both classes equally regardless of prevalence. Unlike positive-class-only F1, Macro F1 drops significantly when the model ignores either class — making it the best early-warning indicator of class-imbalance collapse. |

### When to Use Which Metric

- **Balanced dataset**: All metrics are equally informative. Use F1 or accuracy for quick assessment.
- **Imbalanced dataset** (common in sentiment analysis): Accuracy is misleading. Use **Macro F1** (detects negative-class neglect), **MCC** (robust single-number summary), and **Average Precision** (stricter than AUC-ROC).
- **Negative-class performance matters**: Use **NPV** and **Negative Accuracy** alongside precision/recall.
- **Model ranking quality**: Use **AUC-ROC** (threshold-independent) and **Average Precision** (focuses on positive class).

### Key Literature

- **MCC as best single metric**: Chicco & Jurman (2020), "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," *BMC Genomics* 21:6. — Demonstrates MCC is more informative than F1 and accuracy across binary classification tasks, including imbalanced ones. ([Springer Nature](https://bmcgenomics.biomedcentral.com/article/10.1186/s12864-019-6413-7))
- **MCC real-world challenges**: Zhu & Wang (2023), "Challenges in the real world use of classification accuracy metrics: From recall and precision to the Matthews correlation coefficient," *PLOS One*. — Reviews practical pitfalls of recall/precision/F1 and recommends MCC. ([PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0291908))
- **Macro F1 definition and variants**: Opitz & Burst (2019), "Macro F1 and Macro F1," *arXiv:1911.03347*. — Shows macro F1 weights both classes equally, penalizing classifiers that ignore the minority class. ([arXiv](https://ui.adsabs.harvard.edu/abs/2019arXiv191103347O/abstract))
- **PR-AUC vs ROC-AUC for imbalance**: Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS One*. — Definitive study showing PR curves (and Average Precision) are more informative than ROC curves when classes are imbalanced. ([PMC](https://ncbi.nlm.nih.gov/pmc/articles/PMC4349800/))
- **ROC-AUC still valid for imbalance**: Richardson et al. (2024), "The receiver operating characteristic curve accurately assesses imbalanced datasets," *Patterns* 5(6). — Counterpoint showing ROC-AUC can still be valid, but recommends using it alongside PR-based metrics. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666389924001090))

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
- `sentimentizer_training_val_mcc{model_type}`
- `sentimentizer_training_val_npv{model_type}`
- `sentimentizer_training_val_macro_f1{model_type}`
- `sentimentizer_training_val_auc_roc{model_type}`
- `sentimentizer_training_val_avg_precision{model_type}`
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
- `ray_sentimentizer_live_val_mcc{model_type}`
- `ray_sentimentizer_live_val_npv{model_type}`
- `ray_sentimentizer_live_val_macro_f1{model_type}`
- `ray_sentimentizer_live_val_auc_roc{model_type}`
- `ray_sentimentizer_live_val_avg_precision{model_type}`
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

## Stale Metrics Reset

When training starts for a model type (e.g., `encoder`), any residual metrics from a previous model type's training run (e.g., `rnn`) would persist in two places:

1. **Persisted JSON files** (`/tmp/sentimentizer_metrics/{model_type}_metrics.json`) — each model type writes to its own file, so concurrent training processes never race on a shared file. The standalone exporter reads all three files and updates gauges for each model type independently. Missing files result in zeroed-out gauges.
```
2. **`prometheus_client` gauges** — retain their last-set values in the driver process until overwritten. A Prometheus scrape mid-training would show stale data.
3. **`_RAY_GAUGES` cache** — lazily-created Ray Gauge dicts keyed by `model_type`. If a previous Ray session created gauges for a model type, those cached entries reference gauges from a now-defunct Ray worker context.

**Fix:** `_reset_stale_metrics(model_type)` in `workflows/stages/train.py` is called at the start of `run_train()` (both single-node and distributed paths). It:

1. Writes a **zeroed-out entry** for the current `model_type` in the JSON file (sets all metrics to 0, epoch to 0). This is essential because the standalone exporter only updates gauge labels for model types present in the file — deleting the entry would leave the exporter serving stale values from its in-process gauges.
2. Resets all 11 `sentimentizer_training_*` Prometheus gauges for that `model_type` to `0`.
3. Invalidates the `_RAY_GAUGES` cache entry for that `model_type`, forcing lazy re-creation in the current worker context.

Other model types' entries in the JSON file are left untouched — they'll naturally be updated when those model types are trained next.

## How Training Metrics Flow

### During Distributed Training (`make train-distributed`)

1. **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges every 100 training steps from rank 0
2. **Driver process** sets `sentimentizer_training_*` gauges in-process and writes final metrics to `/tmp/sentimentizer_metrics/{model_type}_metrics.json`
3. **Standalone exporter** (port 8081) reads the JSON file every 10 seconds and updates its gauges
4. **Prometheus** scrapes ports 8080 and 8081 every 10 seconds
5. **Grafana** queries `sentimentizer_training_* or ray_sentimentizer_live_*` to show whichever data is available

### After Training Completes

1. **Driver process** writes final metrics to `/tmp/sentimentizer_metrics/{model_type}_metrics.json`
2. **Driver process** exits
3. **Standalone exporter** (port 8081, always-on) reads the JSON file every 10 seconds and updates its `sentimentizer_training_*` gauges
4. **Prometheus** continues scraping port 8081, picking up the persisted metrics
5. **Grafana** shows the final training results even after training is done

### During Single-Node Training (`make train`)

1. **Trainer** (in the driver process) sets `sentimentizer_training_*` gauges at each validation epoch and writes metrics to JSON
2. **Standalone exporter** (port 8081) reads the JSON file and serves the gauges
3. Same persistence flow as above when training completes

## NaN Handling in Metrics

During training, `torch.sigmoid()` on extreme logit values can produce NaN, which would cause metric computation errors. The codebase handles this at multiple levels:

1. **`sentimentizer/metrics.py`**: `compute_classification_metrics()` replaces NaN probabilities with 0.5 (random-guess probability) via `_replace_nan_probs()` before passing to `torchmetrics.BinaryAUROC`. This also logs a warning for debugging. Cohen's kappa from `torchmetrics.BinaryCohenKappa` returns NaN for single-class targets — `_safe_item()` coerces this to 0.0 to avoid Prometheus gauge issues.
2. **`sentimentizer/metrics.py`**: `compute_metrics_from_model()` replaces NaN at the tensor level using `torch.where()` before converting to numpy
3. **`sentimentizer/trainer.py`**: Both `Trainer.evaluate()` and `_train_func()` (Ray) replace NaN in probabilities with 0.5 after `torch.sigmoid()`

## Starting the Metrics Stack

```bash
# Start Prometheus + Grafana + exporter (regenerates dashboards and restarts Grafana)
make start-metrics

# Or run components individually:
cd metrics && docker compose up -d          # Prometheus + Grafana
make start-exporter                           # Standalone exporter
make setup-dashboards                         # Regenerate dashboard JSON files only
```

**`make start-metrics` does three things**:
1. Runs `make setup-dashboards` — regenerates all dashboard JSON files from `scripts/generate_ray_dashboards.py`
2. Starts Prometheus + Grafana via Docker Compose
3. **Restarts Grafana** so it picks up the freshly generated dashboard files
4. Starts the standalone metrics exporter on port 8081

Grafana only reads provisioned dashboard files on startup, so the restart is required after any dashboard code changes.

### Exporter address (`--addr`)

`sentimentizer/exporter.py` accepts an `--addr` argument (default `127.0.0.1`):

```bash
# Localhost only (default) — not reachable from Docker
uv run python sentimentizer/exporter.py --port 8081 --addr 127.0.0.1

# Listen on all interfaces — required when Prometheus runs in Docker
uv run python sentimentizer/exporter.py --port 8081 --addr 0.0.0.0
```

Prometheus is configured to scrape `host.docker.internal:8081`. On Linux this resolves to the Docker bridge IP (`172.17.0.1`), so the exporter **must** bind to `0.0.0.0` (not `127.0.0.1`) for Prometheus to reach it from within a container.

### Daemonization (`make start-metrics` vs `&`)

The `Makefile` uses `nohup … & disown` (inside a `bash -c` subshell) instead of plain `&`. This is required because `make` spawns a shell that terminates when the target finishes, which sends `SIGHUP` to background jobs and kills them.

- `nohup` catches SIGHUP
- `disown` removes the job from shell tracking entirely
- The process survives after `make start-metrics` exits

### Port conflicts

- Port 8080: Ray metrics (auto-configured by `ray.init(_metrics_export_port=8080)`)
- Port 8081: Standalone exporter (`sentimentizer/exporter.py`)
- Port 8082: Tune metrics (started by `tune_model()`)

If a port is already in use, the code logs a warning and continues — `prometheus_client` gauges are process-global, so metrics still work within the same process.

## Grafana Dashboards

The following dashboards are provisioned in Grafana:

| Dashboard | UID | Metrics Shown |
|-----------|-----|---------------|
| **Model Training** | `sentimentizerModelTraining` | Training loss, validation accuracy / precision / recall / F1, per-class accuracy, Cohen's Kappa, AUC-ROC, epoch |
| **Model Tuning** | `sentimentizerModelTuning` | Ray Tune trial metrics: aggregate stats (best accuracy/loss/F1, trial counts) and per-trial time-series for all validation metrics |
| **Sentimentizer System** | `sentimentizerSystem` | CPU / memory / disk usage, GPU utilization / memory / temperature, Ray cluster health (availability, node count, controller state) |

All custom dashboards use the provisioned Prometheus datasource (`uid: prometheus`).

The Model Training dashboard uses PromQL `or` expressions to fall back between data sources:
```promql
# Shows training metrics if available, falls back to live Ray metrics
sentimentizer_training_val_accuracy{model_type=~"$model_type"}
  or ray_sentimentizer_live_val_accuracy{model_type=~"$model_type"}
```

## Troubleshooting

### Dashboard shows "No data"

1. **Check Prometheus targets**: Visit `http://localhost:9090/targets` — `sentimentizer` and `ray` should be UP
2. **Check exporter is running**: `curl http://localhost:8081/metrics | grep sentimentizer_training` should show metrics for models that have completed training
3. **Check training completed**: `cat /tmp/sentimentizer_metrics/{model_type}_metrics.json` should contain the final metrics
4. **Check dashboard queries**: The dashboard uses `ray_` prefix for Ray metrics (`ray_sentimentizer_live_*`), not `sentimentizer_live_*`

### Metrics disappear after training ends

The standalone exporter (port 8081) reads persisted metrics from `/tmp/sentimentizer_metrics/{model_type}_metrics.json` every 10 seconds for each model type. If a file doesn't exist or is empty, the corresponding model_type gauges will be zeroed out.

### "ValueError: Input contains NaN" crash

This has been fixed. NaN values in probabilities (from extreme logit values) are now replaced with 0.5 at multiple levels. See the "NaN Handling" section above.

### Exporter shows `up=0` in Prometheus or `Connection refused`

This is usually one of three problems:

1. **Exporter launched without `--addr 0.0.0.0`**  
   By default `sentimentizer/exporter.py` binds to `127.0.0.1` (localhost only). Prometheus runs inside Docker and reaches the host via `host.docker.internal`, which resolves to the Docker bridge IP (`172.17.0.1`). If the exporter is on `127.0.0.1`, the container sees `Connection refused`.  
   **Fix:** Start with `--addr 0.0.0.0`:
   ```bash
   make start-exporter   # (Makefile already passes --addr 0.0.0.0)
   ```

2. **Exporter dies when `make` exits**  
   `make` spawns a shell that terminates when the target finishes. Background `&` jobs inside that shell receive `SIGHUP` and die.  
   **Fix:** The `Makefile` now uses `nohup … & disown` inside a `bash -c` subshell so the exporter survives the shell exit.

3. **`make start-metrics` terminates before reaching "All metrics services running"**  
   `pkill -f "sentimentizer/exporter.py"` matched **itself** (its own command line contains `"sentimentizer/exporter.py"`), so it killed the shell running `make`.  
   **Fix:** The kill pattern now uses the bracket trick:  
   ```bash
   pgrep -f "[s]entimentizer/exporter.py"
   ```  
   The `[s]` is a regex character class matching the letter `s`, but the literal `[s]` in `pgrep`'s own argv doesn't match the pattern, so `pgrep` filters itself out.

### Stale metrics from previous training run appear on dashboard

If you train the RNN model and then train the encoder model, the dashboard may briefly show RNN metrics alongside encoder metrics. This happens because `prometheus_client` gauges retain their last-set values until explicitly overwritten.

**This is now handled automatically.** `_reset_stale_metrics(model_type)` is called at the start of every training run and:
1. Writes a zeroed-out per-model JSON file (`/tmp/sentimentizer_metrics/{model_type}_metrics.json`) with a `_trace` field documenting the reset, so the exporter clear stale values.
2. Resets all 11 `sentimentizer_training_*` Prometheus gauges for that `model_type` to `0`.
3. Invalidates the Ray gauge cache for that `model_type`.

Each model type's metrics live in its own file, so concurrent training processes never race.

If you still see stale data, check:

1. Is the standalone exporter running? It polls individual files every 10 seconds.
2. Did the exporter start before the training run? If not, restart the exporter.
3. Check the JSON files: `ls /tmp/sentimentizer_metrics/*.json` — each file should correspond to a model type; the one currently being trained should have `_trace.reset_by` matching its filename.

### Ray metrics not appearing

Ray metrics are only available while Ray is running. Check that `ray` is in the UP state at `http://localhost:9090/targets`. The `ray_sentimentizer_live_*` gauges are set by rank 0 workers every 100 training steps.

### Grafana PromQL Parse Errors (e.g., `unexpected ","`)

If you encounter syntax or parse errors in Grafana after modifying the dashboard generator scripts, check the following common pitfalls:

1. **Invalid `label_values()` macro usage**: Grafana's Prometheus datasource plugin does **not** support combining multiple `label_values()` macro calls using `+` or `OR` in template variable queries (e.g., `label_values(metric_A, label) + label_values(metric_B, label)`). This will crash Grafana's PromQL parser. The correct format is to use a single `label_values()` call referencing a highly reliable metric (such as `ray_node_cpu_count` which is emitted universally) instead of trying to chain fragile metrics.
2. **Hanging commas from stripped filters**: Ray's internal dashboard generators use Jinja2-style placeholders (like `{{{global_filters}}}`). When scripts strip these placeholders out during patching, it can leave behind leading commas (`{, label="x"}`) or consecutive commas (`{x="1", , y="2"}`). The `scripts/generate_ray_dashboards.py` includes robust regex sanitization to clean these artifacts natively.