# Nightly Retraining Scheduler — Recommendations

## Context

The sentimentizer pipeline already has:
- A **Dockerfile** (CPU-only, Ray Serve inference)
- **K8s manifests** in `k8s/` (Deployment, Service, HPA, PDB, Ingress) — serving only
- **GitHub Actions CI** (`ci.yaml`) — lint + test on push/PR
- **HuggingFace Hub integration** — `make push-hub` / `make pull-hub` for weight sync
- **Prometheus/Grafana metrics stack** — persisted JSON + exporter on port 8081
- **Makefile targets** — `train-rnn`, `train-encoder`, `train-decoder`, `push-hub`

The goal is to run **nightly retraining** (extract → tokenize → train → push weights) on a schedule, with the ability to deploy on Docker or K8s.

---

## Option 1: Kubernetes CronJob ⭐ Recommended for K8s

**What it is**: A native K8s resource that runs a Pod on a cron schedule, then terminates.

**Why it fits**: You already have K8s manifests. CronJobs are the idiomatic K8s answer to scheduled batch workloads — no extra dependencies, no new control plane.

### Implementation Sketch

```yaml
# k8s/cronjob-train.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: sentimentizer-nightly-train
  labels:
    app: sentimentizer
    component: training
spec:
  schedule: "0 3 * * *"          # 3 AM UTC nightly
  concurrencyPolicy: Forbid      # Skip if previous run still going
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 5
  startingDeadlineSeconds: 600   # Give up if 10 min late
  jobTemplate:
    spec:
      backoffLimit: 2            # Retry twice on failure
      activeDeadlineSeconds: 14400  # Kill after 4 hours
      template:
        metadata:
          labels:
            app: sentimentizer
            component: training
        spec:
          restartPolicy: Never
          containers:
          - name: trainer
            image: sentimentizer:latest   # Build a training variant
            command: ["sentimentizer"]
            args:
              - "--model"
              - "encoder"
              - "--device"
              - "cpu"
              - "--run-type"
              - "update"
              - "run"
              - "--stop"
              - "300000"
              - "--save"
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
            volumeMounts:
            - name: data-volume
              mountPath: /app/sentimentizer/data
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: sentimentizer-data
```

### What you'd need to add
1. **Training Dockerfile** — Extend the existing one to include data download + training CLI instead of `serve run`
2. **PVC** for `/app/sentimentizer/data` (dictionary, parquet, weights persist across runs)
3. **K8s Secret** for `HF_TOKEN` (for `hf push` after training)
4. **Post-train step** — Either a sidecar/init container or a multi-step command that runs `sentimentizer hf push` after training

### Pros
- Native K8s, no extra tooling
- `concurrencyPolicy: Forbid` prevents overlapping runs
- `backoffLimit` gives automatic retries
- Pod logs visible via `kubectl logs`
- Works with GPU nodes (add `nvidia.com/gpu: 1` resource request + tolerations)

### Cons
- No built-in DAG orchestration (extract → tokenize → train → push is a single linear command)
- Monitoring requires log aggregation or custom alerting
- Debugging failed jobs less ergonomic than a web UI

---

## Option 2: GitHub Actions Scheduled Workflow ⭐ Recommended to start

**What it is**: A `schedule` trigger in GitHub Actions with a `cron` expression.

**Why it fits**: You already have `ci.yaml`. This is zero-infra — GitHub hosts the runner. Good for CPU-only training or as a triggering mechanism.

### Implementation Sketch

```yaml
# .github/workflows/nightly-train.yaml
name: nightly-retrain

on:
  schedule:
    - cron: "0 3 * * *"       # 3 AM UTC
  workflow_dispatch:            # Allow manual trigger

jobs:
  retrain:
    name: nightly retraining
    runs-on: ubuntu-latest      # Or self-hosted for GPU
    timeout-minutes: 240

    steps:
    - uses: actions/checkout@v4

    - name: set up Python 3.12
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: install uv
      uses: astral-sh/setup-uv@v4
      with:
        enable-cache: true

    - name: install dependencies (CPU-only)
      run: |
        uv pip install --index-url https://download.pytorch.org/whl/cpu torch
        uv sync --extra ray

    - name: pull current weights
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        uv run sentimentizer --model encoder hf pull

    - name: download training data
      run: bash scripts/download_data.sh

    - name: run training pipeline
      run: |
        uv run sentimentizer --model encoder --device cpu \
          --run-type update run --stop 300000 --save

    - name: push updated weights
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        uv run sentimentizer --model encoder hf push

    - name: trigger serving deployment update
      if: success()
      run: |
        # Option A: kubectl rollout restart
        # Option B: Trigger a separate deploy workflow
        # Option C: HF webhook auto-deploys
        echo "Weights pushed. Trigger deploy if needed."
```

### Pros
- **Zero infrastructure** — no cluster, no Docker registry, no PVCs
- Already familiar from `ci.yaml`
- `workflow_dispatch` gives manual re-run capability
- Built-in email/Slack notifications on failure
- Artifact upload for training logs/metrics

### Cons
- **6-hour max** on GitHub-hosted runners (enough for CPU training of 300K rows, tight for GPU-scale)
- **No persistent storage** — must pull weights from HuggingFace at start, push at end
- **No GPU** unless you use self-hosted runners
- Scheduled runs can be delayed up to 15 min by GitHub

---

## Option 3: Argo Workflows (K8s-native DAG orchestrator)

**What it is**: A K8s CRD that adds full DAG workflow support — each step is a Pod, with dependencies, retries, artifacts, and a web UI.

**Why it fits**: If the pipeline grows more complex (multi-model, A/B evaluation, conditional rollout), Argo gives first-class support.

### Implementation Sketch

```yaml
# k8s/argo/nightly-train-workflow.yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: sentimentizer-nightly
spec:
  schedule: "0 3 * * *"
  concurrencyPolicy: Forbid
  workflowSpec:
    entrypoint: train-pipeline
    templates:
    - name: train-pipeline
      dag:
        tasks:
        - name: extract
          template: run-stage
          arguments:
            parameters: [{name: cmd, value: "sentimentizer --model encoder extract --stop 300000"}]
        - name: tokenize
          template: run-stage
          dependencies: [extract]
          arguments:
            parameters: [{name: cmd, value: "sentimentizer --model encoder tokenize"}]
        - name: train
          template: run-stage
          dependencies: [tokenize]
          arguments:
            parameters: [{name: cmd, value: "sentimentizer --model encoder --device cpu train --save"}]
        - name: push-weights
          template: run-stage
          dependencies: [train]
          arguments:
            parameters: [{name: cmd, value: "sentimentizer --model encoder hf push"}]

    - name: run-stage
      inputs:
        parameters:
        - name: cmd
      container:
        image: sentimentizer:latest
        command: ["/bin/sh", "-c"]
        args: ["{{inputs.parameters.cmd}}"]
        volumeMounts:
        - name: data
          mountPath: /app/sentimentizer/data
```

### Pros
- **DAG orchestration** — extract/tokenize/train/push as discrete steps with dependencies
- **Web UI** — visual pipeline status, logs per step, retry individual steps
- **Artifact passing** between steps
- **Conditional logic** — e.g., skip push if validation accuracy < threshold
- Supports GPU scheduling per-step (only the train step needs GPU)

### Cons
- **Extra CRD installation** (`kubectl apply` the Argo controller + server)
- Overkill for a 4-step linear pipeline right now
- Learning curve for Argo template syntax
- Adds operational burden (Argo controller resource usage, upgrades)

---

## Option 4: Cron + Docker Compose (Lightweight / single-server)

**What it is**: A system cron job on your dev/training server that runs a training container via Docker Compose.

**Why it fits**: Simplest option if you're running on a single GPU server and don't need K8s.

### Implementation Sketch

```yaml
# docker-compose.train.yaml
services:
  trainer:
    build:
      context: .
      dockerfile: Dockerfile.train
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./sentimentizer/data:/app/sentimentizer/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      bash -c "
        sentimentizer --model encoder --device auto --run-type update run --stop 300000 --save &&
        sentimentizer --model encoder hf push
      "
```

```bash
# /etc/cron.d/sentimentizer-nightly
0 3 * * * root cd /opt/sentimentizer && docker compose -f docker-compose.train.yaml up --abort-on-container-exit >> /var/log/sentimentizer-train.log 2>&1
```

### Pros
- **Simplest** — no K8s, no CI, just cron + Docker
- **GPU access** via NVIDIA Container Toolkit
- Local data volume — no PVC complexity
- Logs to a file

### Cons
- Single point of failure (if the server is down, training doesn't run)
- No retry logic (must add wrapper script)
- No web UI for monitoring
- Manual alerting (parse log for errors, send notification)

---

## Comparison Matrix

| Criteria                 | K8s CronJob      | GitHub Actions    | Argo Workflows    | Cron + Docker    |
|--------------------------|------------------|-------------------|-------------------|------------------|
| **Infra required**       | K8s cluster      | None (hosted)     | K8s + Argo CRD    | Single server    |
| **GPU support**          | ✅ (node pools)   | ⚠️ (self-hosted)  | ✅ (node pools)    | ✅ (native)       |
| **DAG orchestration**    | ❌                | ⚠️ (job deps)     | ✅                 | ❌                |
| **Retries**              | ✅ (backoffLimit) | ✅ (built-in)     | ✅ (per-step)      | ❌ (manual)       |
| **Web UI**               | ❌ (kubectl)      | ✅                 | ✅                 | ❌                |
| **Monitoring/Alerts**    | ⚠️ (add-on)      | ✅ (built-in)     | ✅ (built-in)      | ❌ (manual)       |
| **Complexity**           | Low              | Very Low          | Medium            | Very Low         |
| **Existing infra match** | ✅ (k8s/ exists)  | ✅ (ci.yaml)      | ⚠️ (needs install) | ✅ (Dockerfile)   |
| **Cost**                 | Cluster cost     | Free (2K min/mo)  | Cluster cost      | Server cost      |

---

## GCP GPU Deployment

Two practical paths for running nightly GPU training on Google Cloud.

### Shared Prerequisite: Training Dockerfile

Both GCP options need a training-specific image (the existing `Dockerfile` is serving-only):

```dockerfile
# Dockerfile.train
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project
COPY pyproject.toml README.md LICENSE ./
COPY sentimentizer/ ./sentimentizer/
COPY workflows/ ./workflows/
COPY scripts/ ./scripts/

# Install with CUDA-enabled PyTorch
RUN uv pip install --system --no-cache-dir torch && \
    uv pip install --system --no-cache-dir ".[ray]"

# Training entrypoint script
COPY <<'EOF' /app/train-and-push.sh
#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-encoder}"
STOP="${STOP:-300000}"
DEVICE="${DEVICE:-auto}"

echo "==> Starting nightly training: model=$MODEL, stop=$STOP, device=$DEVICE"
echo "==> $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Pull latest weights (if they exist)
sentimentizer --model "$MODEL" hf pull || echo "No existing weights on Hub, starting fresh"

# Run full pipeline: extract → tokenize → train
sentimentizer --model "$MODEL" --device "$DEVICE" --run-type update run --stop "$STOP" --save

# Push updated weights to HuggingFace Hub
sentimentizer --model "$MODEL" hf push

echo "==> Training complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
EOF
RUN chmod +x /app/train-and-push.sh

ENTRYPOINT ["/app/train-and-push.sh"]
```

### GCP Path A: GKE + GPU Node Pool (K8s CronJob)

Best if you're already running (or plan to run) K8s for serving. The training CronJob lives in the same cluster.

#### 1. Create GKE cluster with GPU node pool

```bash
# Create cluster (Autopilot auto-provisions GPUs per pod request)
gcloud container clusters create-auto sentimentizer-cluster \
  --region us-central1 \
  --project YOUR_PROJECT

# OR: Standard mode with explicit GPU node pool that scales to zero
gcloud container clusters create sentimentizer-cluster \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-2

gcloud container node-pools create gpu-pool \
  --cluster sentimentizer-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 0 \
  --enable-autoscaling --min-nodes 0 --max-nodes 1 \
  --spot   # 60-70% discount for batch workloads
```

> [!IMPORTANT]
> `--min-nodes 0` is critical — the GPU node only spins up when the CronJob creates a pod, so you're not paying for idle GPU time.

#### 2. Install NVIDIA GPU drivers

```bash
# GKE auto-installs on GKE ≥1.29, but for older versions:
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

#### 3. Push training image to Artifact Registry

```bash
# Create repo (one-time)
gcloud artifacts repositories create sentimentizer \
  --repository-format=docker \
  --location=us-central1

# Build & push
IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT/sentimentizer/trainer:latest
docker build -t $IMAGE -f Dockerfile.train .
docker push $IMAGE
```

#### 4. Create K8s secrets and PVC

```yaml
# k8s/secret-hf.yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-credentials
type: Opaque
stringData:
  token: "hf_YOUR_TOKEN_HERE"    # Or use: kubectl create secret generic hf-credentials --from-literal=token=hf_...
```

```yaml
# k8s/pvc-data.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sentimentizer-data
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: standard-rwo   # GKE default SSD-backed storage
  resources:
    requests:
      storage: 20Gi               # Yelp dataset + dictionary + weights
```

#### 5. CronJob with GPU

```yaml
# k8s/cronjob-train.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: sentimentizer-nightly-train
  labels:
    app: sentimentizer
    component: training
spec:
  schedule: "0 3 * * *"              # 3 AM UTC
  timeZone: "America/New_York"       # K8s 1.27+ supports IANA timezones
  concurrencyPolicy: Forbid          # Never overlap runs
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 5
  startingDeadlineSeconds: 600
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 14400   # 4-hour safety kill
      template:
        metadata:
          labels:
            app: sentimentizer
            component: training
        spec:
          restartPolicy: Never
          # Schedule onto the spot GPU node pool
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-t4
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
          - key: cloud.google.com/gke-spot
            operator: Equal
            value: "true"
            effect: NoSchedule
          containers:
          - name: trainer
            image: us-central1-docker.pkg.dev/YOUR_PROJECT/sentimentizer/trainer:latest
            env:
            - name: MODEL
              value: "encoder"
            - name: STOP
              value: "300000"
            - name: DEVICE
              value: "auto"       # Will detect T4 GPU
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
                nvidia.com/gpu: 1   # ← Triggers GPU node scale-up
              limits:
                memory: "8Gi"
                cpu: "4"
                nvidia.com/gpu: 1
            volumeMounts:
            - name: data
              mountPath: /app/sentimentizer/data
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: sentimentizer-data
```

#### 6. Deploy

```bash
kubectl apply -f k8s/secret-hf.yaml
kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/cronjob-train.yaml
```

#### 7. Monitor and test

```bash
# Manually trigger a test run
kubectl create job --from=cronjob/sentimentizer-nightly-train test-train-001

# Watch pod status
kubectl get pods -l component=training -w

# Tail logs
kubectl logs -l component=training -f

# Check job history
kubectl get jobs -l app=sentimentizer --sort-by=.metadata.creationTimestamp
```

#### Optional: Auto-restart serving pods after training

Add a post-training step to the entrypoint script, or use a separate Job:

```bash
# Append to train-and-push.sh (requires RBAC for kubectl in the pod)
kubectl rollout restart deployment/sentimentizer
```

Or use a **Kubernetes Job hook** — create a second Job that watches for CronJob completion and triggers the rollout.

#### Cost estimate (GKE)

| Resource | Spec | Monthly cost |
|----------|------|-------------|
| GKE cluster fee | Autopilot or Standard | $0 (Autopilot) or ~$73 (Standard) |
| GPU node (spot T4) | n1-standard-4 + T4, 2hr/night | ~$7-10/mo |
| PVC (20 GiB SSD) | standard-rwo | ~$3/mo |
| **Total** | | **~$10-85/mo** |

---

### GCP Path B: GCE Spot VM + Instance Schedule (no K8s)

Simplest GCP option. A single VM with a GPU that auto-starts nightly, trains, pushes weights, and stops itself.

#### 1. Create the spot GPU VM

```bash
gcloud compute instances create sentimentizer-trainer \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --maintenance-policy TERMINATE \
  --provisioning-model SPOT \
  --boot-disk-size 50GB \
  --image-family pytorch-latest-gpu \
  --image-project deeplearning-platform-release \
  --metadata HF_TOKEN=hf_YOUR_TOKEN_HERE \
  --tags sentimentizer-trainer \
  --no-restart-on-failure \
  --scopes cloud-platform
```

> [!TIP]
> The `deeplearning-platform-release/pytorch-latest-gpu` image comes with CUDA drivers, PyTorch, and Python pre-installed — saves 10+ minutes of setup per run.

#### 2. Create a startup script

```bash
# Upload to GCS
cat > /tmp/nightly-train.sh << 'SCRIPT'
#!/bin/bash
set -euo pipefail

LOG="/var/log/sentimentizer-train.log"
exec > >(tee -a "$LOG") 2>&1

echo "==> VM started: $(date -u)"

# Install sentimentizer (or use a pre-baked image)
cd /opt
if [ ! -d "torch-sentiment" ]; then
  git clone https://github.com/YOUR_USER/torch-sentiment.git
fi
cd torch-sentiment
git pull origin master

# Install deps
pip install uv
uv sync --extra ray

# Get HF token from instance metadata
export HF_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/HF_TOKEN)

# Pull latest weights
uv run sentimentizer --model encoder hf pull || true

# Train
uv run sentimentizer --model encoder --device auto --run-type update run --stop 300000 --save

# Push weights
uv run sentimentizer --model encoder hf push

echo "==> Training complete: $(date -u)"

# Stop the VM to stop billing
# (instance schedule will start it again tomorrow)
sudo shutdown -h now
SCRIPT

gsutil cp /tmp/nightly-train.sh gs://YOUR_BUCKET/nightly-train.sh

# Attach to VM
gcloud compute instances add-metadata sentimentizer-trainer \
  --zone us-central1-a \
  --metadata startup-script-url=gs://YOUR_BUCKET/nightly-train.sh
```

#### 3. Schedule auto-start/stop

```bash
# Create schedule: start at 3 AM ET, safety-stop at 7 AM ET
gcloud compute resource-policies create instance-schedule nightly-train \
  --region us-central1 \
  --vm-start-schedule="0 7 * * *" \
  --vm-stop-schedule="0 11 * * *" \
  --timezone="UTC"
  # Note: 7 AM UTC = 3 AM ET

# Attach schedule to VM
gcloud compute instances add-resource-policies sentimentizer-trainer \
  --resource-policies=nightly-train \
  --zone us-central1-a
```

> [!NOTE]
> The VM's startup script runs `shutdown -h now` when training completes, so it usually stops well before the 7 AM safety window. The schedule's stop time is just a safety net.

#### 4. Monitor

```bash
# Check VM status
gcloud compute instances describe sentimentizer-trainer \
  --zone us-central1-a --format="value(status)"

# SSH in and tail logs
gcloud compute ssh sentimentizer-trainer --zone us-central1-a \
  --command "tail -f /var/log/sentimentizer-train.log"

# Optional: send logs to Cloud Logging
# (the DL VM image has the logging agent pre-installed)
```

#### Optional: Failure alerting

```bash
# Create an alert policy for VM crash / training failure
gcloud monitoring policies create \
  --notification-channels=YOUR_CHANNEL \
  --display-name="Sentimentizer Training Failed" \
  --condition-display-name="VM stopped unexpectedly" \
  --condition-filter='resource.type="gce_instance" AND resource.labels.instance_id="INSTANCE_ID" AND metric.type="compute.googleapis.com/instance/uptime"'
```

#### Cost estimate (GCE spot)

| Resource | Spec | Monthly cost |
|----------|------|-------------|
| Spot T4 VM | n1-standard-4 + T4, ~2hr/night | ~$7/mo |
| Boot disk | 50 GiB standard | ~$2/mo |
| GCS bucket | Startup script | ~$0/mo |
| **Total** | | **~$9/mo** |

---

### GCP Path Comparison

| | GKE + CronJob | GCE Spot VM |
|---|---|---|
| **Best for** | Already running K8s for serving | Just need nightly GPU training |
| **Monthly cost** | ~$10-85 (depends on cluster mode) | ~$9 |
| **Complexity** | Medium (cluster, node pools, PVCs) | Low (one VM, one script) |
| **GPU cost model** | Node auto-scales to 0 between runs | VM stops between runs |
| **Data persistence** | PVC (survives pod restarts) | Local disk (survives VM stops) |
| **Serving integration** | Same cluster, `kubectl rollout restart` | Separate from serving |
| **Failure handling** | `backoffLimit: 2` auto-retries | Manual (or wrapper script) |
| **Spot preemption** | Pod rescheduled automatically | VM terminated, next run tomorrow |

---

## Recommendation: Phased Approach

### Phase 1 — Start with GitHub Actions (this week)

Add `.github/workflows/nightly-train.yaml` with a `schedule` trigger. This gives you:
- Nightly runs with zero new infrastructure
- Automatic HuggingFace weight push
- Email notifications on failure
- Manual `workflow_dispatch` for ad-hoc retraining
- A working pipeline you can iterate on

> [!TIP]
> Your existing `ci.yaml` is nearly a template — the nightly workflow is the same setup steps plus `train` and `hf push`.

### Phase 2 — Graduate to GCP GPU (when CPU training is too slow)

When CPU training becomes the bottleneck, move to GCP with a GPU:

**Start with GCE Spot VM** (Path B above) — it's ~$9/month, no K8s needed:
1. Create the spot T4 VM with `deeplearning-platform-release` image
2. Attach the `nightly-train.sh` startup script
3. Set an instance schedule for nightly auto-start
4. VM trains → pushes weights to HuggingFace → shuts itself down

**Upgrade to GKE CronJob** (Path A above) if you later deploy serving on K8s:
1. `Dockerfile.train` — CUDA-based training image
2. `k8s/cronjob-train.yaml` — CronJob with `nvidia.com/gpu: 1` resource request
3. `k8s/pvc-data.yaml` — PersistentVolumeClaim for training data
4. Spot GPU node pool with `--min-nodes 0` (auto-scales to zero between runs)

### Phase 3 — Argo Workflows (if pipeline complexity grows)

Only consider if you need:
- Multi-model parallel training (RNN + Encoder + Decoder nightly)
- Conditional rollout (only push if accuracy > threshold)
- A/B evaluation before weight swap
- Complex dependency graphs

---

## What to Build First

If you tell me which approach you'd like to start with, I can implement the full configuration. For **GitHub Actions**, I can create the workflow file immediately. For **K8s CronJob**, I'd also create the training Dockerfile and PVC manifest.

### Key decisions needed:
1. **Which model(s)** to retrain nightly? (encoder only, or all three?)
2. **GPU or CPU** for the nightly run?
3. **Where does training data come from?** (Pre-staged on a volume? Downloaded fresh each run?)
4. **Auto-deploy after training?** (Rolling restart of serving pods, or just push weights and let next deploy pick them up?)
