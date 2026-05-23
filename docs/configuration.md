# Model Configuration

Sentimentizer is designed with modularity in mind. All configurations and hyperparameters are managed as Python dataclasses located in `sentimentizer/config.py`.

---

## Configuration Architecture

At the root of the configuration hierarchy is `DriverConfig`, which acts as a container for all other configuration classes:

- **FileConfig**: Input dataset and output weights file paths.
- **EmbeddingsConfig**: Configuration for downloading and mapping pre-trained word embeddings.
- **TokenizerConfig**: Parameters governing vocabulary construction, sequence padding length, and rating mapping.
- **TrainerConfig**: Hyperparameters for training loops, loss functions, learning rate schedules, and optimizations.
- **RNNConfig**, **EncoderConfig**, **DecoderConfig**, **ModernBERTConfig**: Architecture-specific parameters for each model type.
- **HuggingFaceConfig**: Hugging Face Hub metadata.

---

## Model Config Parameters

### RNN Architecture (LSTM)
Defined by `RNNConfig`. Uses a bidirectional LSTM model.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hidden_size` | integer | `256` | Number of features in the LSTM hidden state. |
| `num_layers` | integer | `2` | Number of stacked LSTM layers. |
| `dropout` | float | `0.2` | Dropout probability applied between LSTM layers. |
| `num_classes` | integer | `3` | Classification output shape (3 = negative/neutral/positive). |

### Transformer Encoder Architecture
Defined by `EncoderConfig`. A standard Transformer Encoder classifier.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | integer | `256` | Dimensionality of the input projected embeddings and attention layers. |
| `n_heads` | integer | `4` | Number of multi-head attention heads. |
| `n_layers` | integer | `4` | Number of Transformer Encoder layers stacked. |
| `dropout` | float | `0.2` | Dropout probability applied to attention and feedforward layers. |
| `ff_multiplier` | integer | `4` | feedforward dimension expansion factor (e.g. `d_model * ff_multiplier = 1024`). |
| `num_classes` | integer | `3` | Classification output shape. |

### Transformer Decoder Architecture
Defined by `DecoderConfig`. An Encoder-Decoder model structure.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | integer | `256` | Dimensionality of the model layers. |
| `n_heads` | integer | `4` | Number of attention heads. |
| `n_encoder_layers` | integer | `2` | Number of Transformer Encoder layers. |
| `n_decoder_layers` | integer | `2` | Number of Transformer Decoder layers. |
| `dropout` | float | `0.3` | Dropout probability (higher to prevent overfitting on decoder cross-attention). |
| `ff_multiplier` | integer | `4` | feedforward dimension expansion factor. |
| `num_classes` | integer | `3` | Classification output shape. |

### ModernBERT Architecture
Defined by `ModernBERTConfig`. Adapts the pre-trained `ModernBERT` transformer.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | string | `"answerdotai/ModernBERT-base"` | Backbone Hugging Face repo name. |
| `dropout` | float | `0.1` | Dropout probability applied in classifier head. |
| `num_classes` | integer | `3` | Classification output shape. |
| `max_seq_length` | integer | `512` | Maximum token sequence length for dynamic padding. |
| `freeze_backbone_epochs`| integer | `1` | Number of epochs to freeze the backbone (training only classifier head). |
| `gradient_checkpointing`| boolean | `True` | Saves ~60% VRAM during backpropagation at ~30% compute overhead. |

---

## Configuration Consistency

Sentimentizer validates configuration consistency automatically prior to initialization via the `validate_config_consistency` function:

- **3-Class Classification Check**: If `TokenizerConfig.include_neutral` is `True`, all models' `num_classes` MUST be exactly `3`.
- **Binary Classification Check**: If `TokenizerConfig.include_neutral` is `False`, all models' `num_classes` MUST be exactly `2`.

If an inconsistency is detected, a `ValueError` is raised, preventing training from starting with invalid shape mismatches.
