"""ONNX export workflow stage.

Exports trained sentiment models (RNN, Encoder, Decoder) to ONNX format
with optional INT8 quantization for CPU deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sentimentizer.export_onnx import export_pipeline

logger = logging.getLogger(__name__)


def run_export(
    state: Any,
    model_type: str,
    quantize: bool = True,
    output_dir: str = "onnx_artifacts",
) -> dict:
    """Export a trained model to ONNX format.

    Args:
        state: State object with model type info (unused, for CLI consistency).
        model_type: One of 'rnn', 'encoder', 'decoder'.
        quantize: Whether to apply INT8 quantization.
        output_dir: Directory for ONNX artifacts.

    Returns:
        Dict with paths to all generated artifacts and validation results.
    """
    logger.info(f"Starting ONNX export for {model_type} (quantize={quantize})")

    results = export_pipeline(
        model_type=model_type,
        output_dir=Path(output_dir),
        quantize=quantize,
    )

    if results["validation"]["passed"]:
        logger.info(
            f"Export successful: {model_type} → {results['onnx_path']} "
            f"(max_diff={results['validation']['max_diff']:.6f})"
        )
    else:
        logger.warning(
            f"Export completed with validation warnings: {model_type} "
            f"max_diff={results['validation']['max_diff']:.6f} "
            f"> tolerance={results['validation']['tolerance']}"
        )

    return results
