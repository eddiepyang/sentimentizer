# SetFit Router (Review Categorization)

The SetFit Router is a lightweight review classification module that categorizes incoming restaurant reviews into targeted areas. This allows downstream systems to route reviews to appropriate teams or track issues (e.g. flagging gluten-free queries to the kitchen).

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
# Run augmentation using default configurations (usually 1,500 train, 300 test items)
make router-augment

# Or customize using the CLI
sentimentizer router augment \
  --model "glm5.1" \
  --variations 50 \
  --output "router/data/augmented_reviews.jsonl"
```

To resume an interrupted generation, add the `--resume` flag to skip already-generated seeds and append new variations.

---

### 2. Model Training (SetFit)

SetFit (Sentence Transformer Fine-Tuning) is an efficient framework for few-shot text classification. It first fine-tunes a Sentence Transformer model (`sentence-transformers/all-MiniLM-L6-v2`) using contrastive learning, and then trains a classification head (logistic regression) on the generated embeddings.

```bash
# Run training on the augmented data
make router-train

# Or train via CLI specifying the data file
sentimentizer router train --data router/data/augmented_reviews.jsonl --output-dir models/router
```

---

### 3. Evaluation

Model evaluation measures macro-averaged precision, recall, and F1-score. The target performance threshold for deployment is **>0.90 Macro F1** across all three classes.

```bash
# Evaluate the model
make router-eval

# Or run via CLI
sentimentizer router evaluate --model-path models/router --data router/data/augmented_reviews.jsonl
```

---

### 4. Push to Hugging Face Hub

Once trained and validated, you can publish the SetFit model directory to the Hugging Face Hub.

```bash
# Push model to the Hub
make router-push

# Or push via CLI with a custom repo ID
sentimentizer router push --model-path models/router --repo-id your-username/sentimentizer-router
```

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
- `--model-path`: **Required**. Path to the trained SetFit router model.
- `--data`: Path to evaluation JSONL data.

### `sentimentizer router push`
- `--model-path`: Path to local model directory.
- `--repo-id`: Hugging Face repository ID.
