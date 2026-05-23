# ONNX Export

Sentimentizer supports exporting trained PyTorch models to ONNX (Open Neural Network Exchange) format. ONNX models can be deployed in highly optimized runtimes like ONNX Runtime, offering substantial speedups and lower memory footprints during inference.

---

## Export Commands

Use the `export` subcommand to convert a model checkpoint to an ONNX graph.

```bash
# Export the RNN model with INT8 quantization
sentimentizer export --model rnn --quantize

# Export the Encoder model without quantization
sentimentizer export --model encoder --no-quantize

# Specify a custom output directory
sentimentizer export --model decoder --output-dir custom_onnx_dir
```

By default, ONNX models and configurations are saved to the `onnx_artifacts/` directory.

---

## CLI Options Reference

| Option | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `rnn`, `encoder`, `decoder`, `modernbert` | **Required** | Model type to export. |
| `--quantize` / `--no-quantize` | flag | `--quantize` | Apply INT8 dynamic quantization to the exported graph. |
| `--output-dir` | string | `"onnx_artifacts"` | Target directory where the exported ONNX assets are saved. |

---

## INT8 Quantization

Passing `--quantize` applies **dynamic INT8 quantization** to the exported ONNX model.
- **How it works**: Floating-point weights are converted to 8-bit integers, reducing the file size by approximately 75% (e.g. from ~60MB down to ~15MB for typical embeddings + model parameter sets).
- **Execution Speed**: Quantized models generally execute significantly faster on CPU runtimes by utilizing integer vector instructions.
- **Accuracy**: Quantization introduces minimal loss of accuracy (usually <0.1% change in validation metrics).

---

## Correctness Validation

During the export stage, Sentimentizer automatically performs an end-to-end correctness check:
1. It runs a set of dummy text sequences through the PyTorch model to get logits.
2. It executes the exact same inputs through the newly exported ONNX model.
3. It validates that the **cosine similarity** between the PyTorch logits and ONNX logits is **greater than 0.9999**.

If the validation check fails (which indicates numerical instability or graph differences), the export process raises an error and terminates before writing the outputs.

---

## Architecture Limitations (ModernBERT)

ModernBERT has architectural constraints that prevent standard ONNX export.

> [!WARNING]
> ModernBERT has `SUPPORTS_ONNX = False` configured in its class definition. If you attempt to run:
> ```bash
> sentimentizer export --model modernbert
> ```
> The CLI will reject the request cleanly with a configuration error message stating that ONNX export is not supported for this model type.
