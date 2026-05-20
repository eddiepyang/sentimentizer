# Encoder Architecture & Training Optimization Plan

> **Status: COMPLETED** — May 19, 2026. These optimizations have been implemented across the Encoder, Decoder, and RNN models.
> Current status: All models feature frozen GloVe embeddings (with trainable OOV tokens), proper `padding_idx` masking, and weight decay exclusions for 1-D parameters. Transformer layers use Pre-LN and GELU activations. The Encoder features mean pooling, and the custom LR scheduler correctly decays to `eta_min`.

This document outlines the planned optimizations for the Transformer Encoder model (`sentimentizer/models/encoder.py`) and its training pipeline (`sentimentizer/trainer.py`). 

## 🌟 High-Impact Optimizations

These two items represent the most significant issues in the current pipeline and will yield the largest improvements in model convergence and stability.

> [!IMPORTANT]
> **1. LR Scheduler Math Mismatch (Critical Bug Fix)**
> * **The Issue**: The `_LinearWarmupCosineScheduler` inherits from `LambdaLR`, which scales the `base_lr` by the return value of the lambda function. When the cosine curve finishes, the lambda returns `eta_min` (e.g., `1e-6`). PyTorch computes `1e-4 * 1e-6 = 1e-10`, effectively stalling the learning rate to near-zero instead of the intended absolute minimum of `1e-6`.
> * **The Plan**: Update the lambda function to return a relative multiplier (`eta_min / base_lr`) so that the final multiplied learning rate equals the exact target value.

> [!IMPORTANT]
> **2. Embedding Regularization & OOV Token Handling (Prevents Massive Overfitting)**
> * **The Issue**: The entire GloVe embedding matrix (~2 million parameters) is currently fully trainable, leading to massive overfitting on small data splits. 
> * **The Plan**: Freeze the embedding weights (`requires_grad=False`) to preserve GloVe's spatial semantics. However, we will explicitly leave the Out-Of-Vocabulary (OOV) token (row `N+1`) **trainable**. If we froze the entire matrix blindly, the OOV token would remain a static, noisy random vector. Allowing only the OOV token to train lets the model learn an optimal "average unknown word" representation. We will also initialize the embedding layer with `padding_idx=0` to stop padding representations from drifting.

---

## 🛠️ Medium & Low-Impact Architectural Improvements

These refinements align the model with modern PyTorch best practices and Transformer conventions.

### 3. Exclude Biases and 1D Tensors from Weight Decay
* **The Issue**: `AdamW` currently applies weight decay to all parameters, including layer normalization scales/biases and linear layer biases. This degrades model capacity.
* **The Plan**: Split `model.parameters()` into two groups in `_create_training_components`—one with weight decay applied, and a `no_decay` group for 1D parameters and biases.

### 4. Modernize Transformer Encoder Layer
* **The Issue**: `nn.TransformerEncoderLayer` defaults to Post-Layer Normalization (`norm_first=False`) and ReLU activation, which are sub-optimal for gradient stability.
* **The Plan**: Update the instantiation to use `norm_first=True` (Pre-LN) and `activation="gelu"`. 

### 5. Evaluate Mean Pooling vs. CLS Token
* **The Issue**: Shallow transformers (like this 4-layer model) often struggle to fully aggregate sentence semantics into a single learnable CLS token.
* **The Plan**: Implement an option for mean pooling over non-padding tokens. We will benchmark this against the current CLS token pooling to see if it yields a higher Macro F1.

### 6. Scale Embedding Projections
* **The Issue**: Positional encodings are added directly to the projected embeddings without scaling. Early in training, the positional signal (sinusoids with variance ~0.5) can overpower the semantic embeddings.
* **The Plan**: Scale the output of the linear projection `self.proj` by `math.sqrt(self.d_model)` before adding the positional encodings.

---

## Implementation Steps

1. **Refactor `sentimentizer/trainer.py`**:
   - Fix the `_LinearWarmupCosineScheduler` lambda calculation.
   - Update `_create_training_components` to dynamically split parameters for weight decay.

2. **Refactor `sentimentizer/models/encoder.py`**:
   - Update `nn.Embedding` initialization to include `padding_idx=0`.
   - Freeze the embedding weights, but isolate and re-enable gradients for the OOV index row.
   - Add `norm_first=True` and `activation="gelu"` to the `TransformerEncoderLayer`.
   - Apply `math.sqrt(d_model)` scaling to projected embeddings.

3. **Validation**:
   - Run the training pipeline to ensure no regressions.
   - Monitor the `TRAINING_VAL_MACRO_F1` to observe performance improvements.
   - Ensure the ONNX export `export_onnx.py` still successfully traces the updated embedding layer.
