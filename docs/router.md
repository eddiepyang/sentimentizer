# Intent Router (Review Categorization)

The intent router is a lightweight review classification module that categorizes incoming restaurant reviews into targeted areas. This allows downstream systems to route reviews to appropriate teams or track issues (e.g. flagging gluten-free queries to the kitchen).

> **Not SetFit.** This module was originally built on the `setfit` library, and
> some names still carry that history (`SetFitConfig` is a backward-compatible
> alias for `RouterConfig`). `setfit` is no longer a dependency and is imported
> nowhere. `RouterModel` (`sentimentizer/router/model.py`) is a
> `SentenceTransformer` backbone fine-tuned with contrastive pairs plus an
> sklearn `LogisticRegression` head.

---

## Category Schema

Reviews are classified into three mutually exclusive categories:

- **Dietary**: Mentions of allergies, gluten-free options, celiac requirements, dairy-free alternatives, vegan/vegetarian options, or dietary restrictions.
- **Service**: Mentions of wait times, reservation issues, staff friendliness, ordering issues, table service, customer support, or staff behavior.
- **General**: Ambiance, decor, pricing, menu variety, food taste, overall review quality, location, or general dining experiences.

---

## Workflow Stages

### 1. Data Augmentation (GLM 5.1 via Ollama)

Since hand-labeling thousands of reviews is expensive, Sentimentizer uses a few-shot augmentation pipeline. It starts with 30 golden seed examples (10 per category) defined in `sentimentizer/router/seeds.py` and uses **GLM 5.1** via **Ollama** to generate synthetic variations.

```bash
# Run augmentation using default configurations (30 seeds x 50 variations = ~1,500 items)
make router-augment

# Or customize using the CLI
sentimentizer router augment \
  --model "glm-5.1:cloud" \
  --variations 50 \
  --output "augmented_yelp.jsonl"
```

To resume an interrupted generation, add the `--resume` flag to skip already-generated seeds and append new variations.

---

### 2. Model Training

Training follows the SetFit *method* without the library: it fine-tunes a sentence-transformer backbone (`BAAI/bge-base-en-v1.5` by default, 109M params, 768-dim) with contrastive pairs, then fits an sklearn `LogisticRegression` head on the resulting embeddings. `RouterConfig.num_iterations` (default 20) controls contrastive pairs generated per example.

```bash
# Run training on the augmented data
make router-train

# Or train via CLI specifying the data file
sentimentizer router train --data augmented_yelp.jsonl --output-dir models/router
```

---

### 3. Evaluation

`evaluate_router()` reports a classification report (accuracy, per-class precision/recall/F1), a centroid cosine-similarity matrix, and a tau threshold calibration.

Embedding-separation targets, enforced as `OK`/`HIGH`/`LOW` labels in the log output:

- **Inter-class similarity < 0.65** — categories must not collapse into each other
- **Intra-class similarity > 0.85** — each category must be internally coherent

```bash
# Evaluate the model
make router-evaluate

# Or run via CLI
sentimentizer router evaluate --model-path models/router --data augmented_yelp.jsonl
```

---

### 4. Push to Hugging Face Hub

Once trained and validated, publish the model directory to the Hugging Face Hub.

```bash
# Push model to the Hub
make upload-router

# Or push via CLI with a custom repo ID
sentimentizer router push --model-path models/router --repo-id your-username/sentimentizer-router
```

Run the whole pipeline (augment → train → evaluate) with `make router-pipeline`.

---

## Environment Variable Overrides

Both `RouterConfig` (training) and `AugmentConfig` (augmentation) support env-var overrides on top of `router/config.yaml`. Loading order (highest priority first):

1. Env var (e.g. `SENTIMENTIZER_ROUTER_NUM_EPOCHS=3`)
2. `sentimentizer/router/config.yaml` value
3. Dataclass default

Training (`SENTIMENTIZER_ROUTER_*`):

| Env var | Field | Type |
| :--- | :--- | :--- |
| `SENTIMENTIZER_ROUTER_BASE_MODEL` | `base_model` | str |
| `SENTIMENTIZER_ROUTER_NUM_ITERATIONS` | `num_iterations` | int |
| `SENTIMENTIZER_ROUTER_NUM_EPOCHS` | `num_epochs` | int |
| `SENTIMENTIZER_ROUTER_BATCH_SIZE` | `batch_size` | int |
| `SENTIMENTIZER_ROUTER_MAX_SEQ_LENGTH` | `max_seq_length` | int |
| `SENTIMENTIZER_ROUTER_SEED` | `seed` | int |
| `SENTIMENTIZER_ROUTER_OUTPUT_DIR` | `output_dir` | str (path) |

Augmentation (`SENTIMENTIZER_AUGMENT_*`):

| Env var | Field | Type |
| :--- | :--- | :--- |
| `SENTIMENTIZER_AUGMENT_MODEL` | `model` | str |
| `SENTIMENTIZER_AUGMENT_OLLAMA_URL` | `ollama_url` | str |
| `SENTIMENTIZER_AUGMENT_VARIATIONS_PER_SEED` | `variations_per_seed` | int |
| `SENTIMENTIZER_AUGMENT_OUTPUT_PATH` | `output_path` | str |
| `SENTIMENTIZER_AUGMENT_BATCH_SIZE` | `batch_size` | int |

Invalid values raise `ValueError` at load time with the offending env var name. See [configuration.md](configuration.md#runtime-configuration) for the full multi-domain reference.

---

## Router CLI Options Reference

### `sentimentizer router augment`
- `--output`: Output JSONL file path.
- `--model`: Ollama model name to use.
- `--variations`: Number of variations to generate per seed utterance.
- `--ollama-url`: Ollama API endpoint.
- `--resume`: Resume from existing output file.
- `--config`: Path to router config YAML (defaults to `router/config.yaml`).

### `sentimentizer router train`
- `--data`: **Required**. Path to the augmented JSONL dataset.
- `--base-model`: Sentence-transformer base model.
- `--output-dir`: Output directory for the trained model.
- `--config`: Path to router config YAML.

### `sentimentizer router evaluate`
- `--model-path`: **Required**. Path to the trained router model directory.
- `--data`: Path to evaluation JSONL data.

### `sentimentizer router push`
- `--model-path`: Path to local model directory.
- `--repo-id`: Hugging Face repository ID.
