# Model Synchronization (Hugging Face Hub)

This document covers how to synchronize, download, and publish model weights and configurations to and from the Hugging Face Hub.

## Pre-trained Repositories

The following default repositories are used for default weights and backbones:

| Model Type | Purpose | Repository ID |
| :--- | :--- | :--- |
| **ModernBERT** | Pre-trained Backbone | `nomic-ai/modernbert-base` |
| **RNN** | Default RNN weights | `eddiepyang/sentimentizer-rnn` |
| **Encoder** | Default Encoder weights | `eddiepyang/sentimentizer-encoder` |
| **Decoder** | Default Decoder weights | `eddiepyang/sentimentizer-decoder` |
| **Router** | Default SetFit routing weights | `eddiepyang/sentimentizer-router` |

---

## Pushing to the Hub

You can publish trained model weights, configuration files, and tokenizers to a Hugging Face repository.

### Triggering Automatically on Train
When you pass `--push-to-hub` (and `--save`) during training, Sentimentizer automatically uploads checkpoints to the Hub upon successful training completion:

```bash
sentimentizer train --save --push-to-hub --hf-repo your-username/sentimentizer-rnn
```

> [!IMPORTANT]
> The `--push-to-hub` flag is ignored if you specify `--no-save`. Checkpoint saving must be enabled to push.

### Explicit Push Command
To push the current local weights for a model type directly to the Hub without training, run:

```bash
# Push the local RNN weights to the Hub
sentimentizer --model rnn hf push --repo-id your-username/sentimentizer-rnn

# Or via Makefile
make push-hub MODEL=rnn HF_REPO=your-username/sentimentizer-rnn
```

---

## Pulling from the Hub

You can pull model weights and configuration sidecars down from a Hugging Face repository to seed your local training or run inference.

### Triggering Automatically on Train
When you pass `--pull-from-hub` to the train command, Sentimentizer downloads the latest weights from the Hub prior to commencing training. This automatically sets `--run-type update` to leverage the downloaded checkpoints:

```bash
sentimentizer train --pull-from-hub --hf-repo your-username/sentimentizer-rnn
```

### Explicit Pull Command
To download model weights directly from the Hub, use the `hf pull` subcommand:

```bash
# Pull RNN weights to your local machine
sentimentizer --model rnn hf pull --repo-id your-username/sentimentizer-rnn

# Or via Makefile
make pull-hub MODEL=rnn HF_REPO=your-username/sentimentizer-rnn
```

---

## Auto-Generated Model Cards

When a model is pushed to the Hugging Face Hub, Sentimentizer automatically generates and uploads a descriptive model card (`README.md` in the repository).

The generated model card includes:
- **Model Description**: The architecture (RNN, Transformer Encoder, Transformer Decoder, or ModernBERT) and training context.
- **Hyperparameter Specifications**: Learning rate, loss type, weight decay parameters, smoothing factors, and optimization configurations.
- **Metrics Log**: Balanced accuracy, macro/weighted F1 scores, precision, recall, and Matthews correlation coefficient (MCC).
- **Inference Instructions**: Python code snippets for loading and running the model with `SentimentPredictor`.
- **Grafana/Prometheus Dashboard Link**: A pointer to the monitoring dashboard setup for model performance logging.
