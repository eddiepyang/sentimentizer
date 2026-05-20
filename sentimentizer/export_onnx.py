"""Export trained sentiment models to ONNX format with INT8 quantization.

Supports RNN, Encoder, and Decoder models. The RNN model uses a masked
fallback path (via _RNNOnnxWrapper) to bypass pack_padded_sequence, which
is incompatible with ONNX tracing.

Usage:
    from sentimentizer.export_onnx import export_pipeline

    results = export_pipeline("rnn", quantize=True)
    results = export_pipeline("encoder", quantize=True)
    results = export_pipeline("decoder", quantize=True)
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from sentimentizer.config import LABEL_NAMES, NUM_CLASSES, weights_path_for
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.rnn import RNN

logger = logging.getLogger(__name__)

ONNX_OPSET_VERSION = 17  # stable, well-tested; opset 18+ requires dynamo_export


class _RNNOnnxWrapper(nn.Module):
    """Wraps RNN to call forward(onnx_export=True) for ONNX tracing.

    torch.onnx.export(model, args, f) calls model(*args) internally.
    We cannot pass onnx_export=True as a keyword argument through the
    standard export call, so this wrapper forces the ONNX-compatible
    forward path (skipping pack_padded_sequence).
    """

    def __init__(self, rnn: RNN) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rnn.forward(inputs, onnx_export=True)


def load_model_for_export(model_type: str, device: str = "cpu") -> nn.Module:
    """Load a trained model for ONNX export (on CPU, in eval mode).

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        device: Device to load weights onto (use 'cpu' for ONNX export).

    Returns:
        Model in eval mode with dropout disabled.

    Raises:
        FileNotFoundError: If trained weights are not found locally or on HF Hub.
        ValueError: If model_type is not recognized.
    """
    if model_type == "rnn":
        model = RNN.get_trained_model(device)
    elif model_type == "encoder":
        model = Encoder.get_trained_model(device)
    elif model_type == "decoder":
        model = Decoder.get_trained_model(device)
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Must be 'rnn', 'encoder', or 'decoder'."
        )

    model.eval()
    return model


def export_model_to_onnx(
    model: nn.Module,
    model_type: str,
    output_path: Path,
    seq_len: int = 200,
    opset_version: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export a trained sentiment model to ONNX format.

    For RNN models, wraps in _RNNOnnxWrapper to bypass pack_padded_sequence.

    Handles:
    - Dynamic batch and sequence length axes
    - RNN masked fallback (via _RNNOnnxWrapper)
    - Encoder/Decoder padding masks (derived internally from input == 0)

    Args:
        model: Trained model in eval mode.
        model_type: One of 'rnn', 'encoder', 'decoder'.
        output_path: Path to write the ONNX model file.
        seq_len: Maximum sequence length for the dummy input.
        opset_version: ONNX opset version (17 recommended).

    Returns:
        Path to the exported ONNX model file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For RNN, wrap to enable ONNX-compatible forward path
    export_model = _RNNOnnxWrapper(model) if model_type == "rnn" else model

    # Create dummy input (token IDs, zero-padded)
    # RNN export works best with batch size 1 to avoid LSTM initial state shape assumptions
    batch_size = 1 if model_type == "rnn" else 2
    dummy_input = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)

    # Move model to CPU for export
    export_model = export_model.cpu()

    logger.info(f"Exporting {model_type} model to ONNX (opset {opset_version})...")

    if model_type == "rnn":
        torch.onnx.export(
            export_model,
            (dummy_input,),
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    else:
        dynamic_shapes = {
            "inputs": {
                0: torch.export.Dim.DYNAMIC,
                1: torch.export.Dim.DYNAMIC,
            }
        }
        torch.onnx.export(
            export_model,
            (dummy_input,),
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
            do_constant_folding=True,
            dynamo=True,
        )

    logger.info(f"Exported {model_type} to {output_path}")
    return output_path


def quantize_onnx_model(
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Apply INT8 dynamic quantization for CPU deployment (AVX-512 optimized).

    Uses onnxruntime.quantization.quantize_dynamic which quantizes weights
    to INT8 while keeping activations in FP32 — optimal for Zen 5 AVX-512.

    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Path to write the quantized ONNX model.
            If None, uses input_path with '_quantized' suffix.

    Returns:
        Path to the quantized ONNX model file.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_quantized.onnx")
    output_path = Path(output_path)

    logger.info(f"Quantizing {input_path} → {output_path} (INT8 dynamic)...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    logger.info(f"Quantized model saved to {output_path}")
    return output_path


def validate_onnx_export(
    onnx_path: Path,
    model: nn.Module,
    model_type: str,
    test_input: torch.Tensor | None = None,
    tolerance: float | None = None,
) -> dict:
    """Verify ONNX model outputs match PyTorch within tolerance.

    For RNN models, the tolerance is relaxed to 1e-2 because the ONNX
    path (masked fallback) produces slightly different numerics than the
    pack_padded_sequence path. Encoder/Decoder models use 1e-4.

    Args:
        onnx_path: Path to the ONNX model file.
        model: Original PyTorch model (used for comparison).
        model_type: One of 'rnn', 'encoder', 'decoder'.
        test_input: Tensor to run through both models.
            If None, generates a random input.
        tolerance: Maximum absolute difference allowed.
            If None, uses 1e-2 for RNN, 1e-4 for others.

    Returns:
        Dict with 'max_diff', 'mean_diff', 'passed' keys.
    """
    import onnxruntime as ort

    if tolerance is None:
        tolerance = 1e-2 if model_type == "rnn" else 1e-4

    if test_input is None:
        test_input = torch.randint(1, 100, (4, 200), dtype=torch.long)

    # PyTorch inference
    model.eval()
    with torch.no_grad():
        if model_type == "rnn":
            pytorch_output = model(test_input, onnx_export=True).cpu().numpy()
        else:
            pytorch_output = model(test_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(
        ["logits"],
        {"input": test_input.cpu().numpy()},
    )[0]

    # Compare
    max_diff = abs(pytorch_output - onnx_output).max()
    mean_diff = abs(pytorch_output - onnx_output).mean()
    passed = max_diff < tolerance

    logger.info(
        f"ONNX validation for {model_type}: max_diff={max_diff:.6f}, "
        f"mean_diff={mean_diff:.6f}, tolerance={tolerance}, passed={passed}"
    )

    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "tolerance": tolerance,
        "passed": passed,
    }


def _save_metadata(
    model_type: str,
    onnx_path: Path,
    quantized_path: Path | None,
    validation: dict,
    opset_version: int,
    seq_len: int,
) -> Path:
    """Save metadata JSON alongside ONNX model artifacts.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        onnx_path: Path to the FP32 ONNX model.
        quantized_path: Path to the quantized ONNX model (or None).
        validation: Validation results dict.
        opset_version: ONNX opset version used.
        seq_len: Maximum sequence length.

    Returns:
        Path to the metadata JSON file.
    """
    metadata = {
        "model_type": model_type,
        "num_classes": NUM_CLASSES,
        "label_names": LABEL_NAMES,
        "opset_version": opset_version,
        "input_shape": ["batch_size", "seq_len"],
        "input_name": "input",
        "output_name": "logits",
        "seq_len": seq_len,
        "dictionary_path": weights_path_for(model_type).replace("_weights.pth", ".dictionary"),
        "quantized": quantized_path is not None,
        "validation": validation,
    }

    metadata_path = onnx_path.with_name(f"{model_type}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata saved to {metadata_path}")
    return metadata_path


def export_pipeline(
    model_type: str,
    output_dir: Path = Path("onnx_artifacts"),
    quantize: bool = True,
    device: str = "cpu",
    seq_len: int = 200,
    opset_version: int = ONNX_OPSET_VERSION,
) -> dict:
    """Full export pipeline: load → export → quantize → validate.

    Creates output_dir if it doesn't exist. Saves metadata JSON alongside
    each ONNX model with model_type, opset_version, input_shape, dictionary
    path, and quantization status.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        output_dir: Directory for ONNX artifacts (auto-created).
        quantize: Whether to apply INT8 quantization.
        device: Device to load model onto (use 'cpu' for export).
        seq_len: Maximum sequence length for dummy input.
        opset_version: ONNX opset version (17 recommended).

    Returns:
        Dict with paths to all generated artifacts and validation results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model
    logger.info(f"Loading {model_type} model for export...")
    model = load_model_for_export(model_type, device)

    # Step 2: Export to ONNX
    onnx_path = output_dir / f"{model_type}.onnx"
    export_model_to_onnx(model, model_type, onnx_path, seq_len, opset_version)

    # Step 3: Quantize (optional)
    quantized_path = None
    if quantize:
        quantized_path = quantize_onnx_model(onnx_path)

    # Step 4: Validate
    # Validate against the quantized model if quantization was applied
    validate_path = quantized_path if quantized_path else onnx_path
    validation = validate_onnx_export(validate_path, model, model_type)

    if not validation["passed"]:
        logger.warning(
            f"ONNX validation FAILED for {model_type}: "
            f"max_diff={validation['max_diff']:.6f} > tolerance={validation['tolerance']}"
        )
    else:
        logger.info(f"ONNX validation PASSED for {model_type}")

    # Step 5: Save metadata
    metadata_path = _save_metadata(
        model_type, onnx_path, quantized_path, validation, opset_version, seq_len
    )

    return {
        "model_type": model_type,
        "onnx_path": str(onnx_path),
        "quantized_path": str(quantized_path) if quantized_path else None,
        "metadata_path": str(metadata_path),
        "validation": validation,
    }
