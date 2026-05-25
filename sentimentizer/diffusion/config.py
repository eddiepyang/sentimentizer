"""Configuration dataclasses for diffusion models.

Each model variant has its own defaults for inference parameters
(steps, guidance_scale, pixel limits, dimension alignment).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionModelConfig:
    """Configuration for a single diffusion model.

    Attributes:
        model_id: HuggingFace model ID or local path for from_pretrained.
        model_path: Path to weights file (e.g. GGUF for FLUX).
        dtype: Torch dtype for inference (e.g. torch.bfloat16).
        quantization: Quantization config string (e.g. "Q8_0") or None.
        default_steps: Default number of denoising steps.
        default_guidance: Default classifier-free guidance scale.
        max_pixels: Maximum width*height allowed (e.g. 1048576 for 1024²).
        dim_alignment: Dimension alignment requirement (8 for SD, 16 for FLUX).
    """

    model_id: str = ""
    model_path: str = ""
    dtype: str = "bfloat16"
    quantization: str | None = None
    default_steps: int = 30
    default_guidance: float = 7.5
    max_pixels: int = 1048576
    dim_alignment: int = 8


SD_DEFAULT_CONFIG = DiffusionModelConfig(
    model_id="stabilityai/stable-diffusion-2-1",
    dtype="bfloat16",
    quantization=None,
    default_steps=30,
    default_guidance=7.5,
    max_pixels=1048576,
    dim_alignment=8,
)

FLUX_DEFAULT_CONFIG = DiffusionModelConfig(
    model_id="black-forest-labs/FLUX.1-dev",
    model_path="",
    dtype="bfloat16",
    quantization="Q8_0",
    default_steps=28,
    default_guidance=3.5,
    max_pixels=1048576,
    dim_alignment=16,
)

SD35_DEFAULT_CONFIG = DiffusionModelConfig(
    model_id="stabilityai/stable-diffusion-3.5-medium",
    dtype="bfloat16",
    quantization=None,
    default_steps=40,
    default_guidance=4.5,
    max_pixels=1048576,
    dim_alignment=16,
)
