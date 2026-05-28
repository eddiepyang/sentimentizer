"""MLX-based diffusion predictors for Apple Silicon."""

from __future__ import annotations

from typing import Any

import PIL.Image

from sentimentizer import logger
from sentimentizer.diffusion.config import FLUX2_KLEIN_DEFAULT_CONFIG, DiffusionModelConfig
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE

_MFLUX_QUANTIZE_MAP: dict[str | None, int | None] = {
    None: None,
    "": None,
    "nf4": 4,
    "int4": 4,
    "4bit": 4,
    "int8": 8,
    "8bit": 8,
}

_MAX_SEED = 2**32 - 1


class MLXFlux2KleinPredictor:
    """FLUX.2 Klein predictor using mflux (MLX) on Apple Silicon.

    ~4-5x faster than diffusers/MPS for 1024x1024 generation.
    Supports 4-bit and 8-bit quantization natively (no bitsandbytes).
    """

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        if not MFLUX_AVAILABLE:
            raise ImportError(
                "mflux is required for the MLX backend. "
                "Install with: pip install sentimentizer[mlx-diffusion]"
            )
        self.cfg = cfg or FLUX2_KLEIN_DEFAULT_CONFIG
        self._device = "mlx"
        self._model_loaded = False
        self._model_error: str | None = None
        self._pipeline: Any = None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_error(self) -> str | None:
        return self._model_error

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

            quant_key = self.cfg.quantization.lower() if self.cfg.quantization else None
            quantize = _MFLUX_QUANTIZE_MAP.get(quant_key)
            if quantize is None and self.cfg.quantization:
                logger.warning(
                    "Unknown quantization=%r for MLX backend; ignoring",
                    self.cfg.quantization,
                )

            model_config = ModelConfig.flux2_klein_4b()
            self._pipeline = Flux2Klein(
                quantize=quantize,
                model_config=model_config,
            )
            self._model_loaded = True
            logger.info(
                "MLX FLUX.2 Klein model warmed up",
                model_id=self.cfg.model_id,
                device=self._device,
                quantization=self.cfg.quantization,
            )
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("MLX FLUX.2 Klein warmup failed")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        reference_images: list[PIL.Image.Image] | None = None,
    ) -> tuple[Any, int]:
        # reference_images not yet supported for MLX backend (v2).
        # Flux2KleinEdit has a different API (image_paths, not PIL.Image).
        if reference_images is not None:
            raise NotImplementedError(
                "reference_images are not yet supported by the MLX backend. "
                "Use backend='diffusers' or set "
                "SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND=diffusers."
            )

        if not self._model_loaded:
            raise RuntimeError(f"MLX FLUX.2 Klein model not loaded: {self._model_error}")

        used_seed = self._resolve_seed_mlx(seed)

        # FLUX.2 Klein is unguided; negative_prompt is silently dropped.
        # (Same behavior as the diffusers Flux2KleinPredictor.)
        del negative_prompt

        if guidance_scale and guidance_scale > 0:
            logger.debug(
                "MLX FLUX.2 Klein ignores guidance_scale=%s (unguided model)",
                guidance_scale,
            )

        try:
            result = self._pipeline.generate_image(
                seed=used_seed,
                prompt=prompt,
                num_inference_steps=(steps if steps is not None else self.cfg.default_steps),
                width=width,
                height=height,
            )
            # result is a GeneratedImage; .image gives PIL.Image
            return result.image, used_seed
        finally:
            # MLX unified-memory cleanup — analogous to gc.collect() +
            # torch.cuda.empty_cache() in the diffusers Klein predictor.
            # Without this, repeated generations balloon resident memory.
            try:
                import mlx.core as mx

                # mx.clear_cache() (>=0.18) or mx.metal.clear_cache() (older).
                clear = getattr(mx, "clear_cache", None) or getattr(
                    getattr(mx, "metal", None), "clear_cache", None
                )
                if clear is not None:
                    clear()
            except ImportError:
                pass

    def _resolve_seed_mlx(self, seed: int | None) -> int:
        """Seed resolution without torch.Generator (uses os.urandom)."""
        if seed is not None:
            if not (0 <= seed <= _MAX_SEED):
                raise ValueError(f"seed must be 0..{_MAX_SEED}, got {seed}")
            return seed
        import os

        return int.from_bytes(os.urandom(4), "big") % (_MAX_SEED + 1)

    def resolve_defaults(self, request: Any) -> dict[str, Any]:
        """Fill in per-model defaults for unset request fields."""
        resolved: dict[str, Any] = {}
        resolved["prompt"] = request.prompt
        resolved["negative_prompt"] = getattr(request, "negative_prompt", None)
        resolved["steps"] = request.steps if request.steps is not None else self.cfg.default_steps
        resolved["guidance_scale"] = (
            request.guidance_scale
            if request.guidance_scale is not None
            else self.cfg.default_guidance
        )
        resolved["width"] = request.width
        resolved["height"] = request.height
        resolved["seed"] = request.seed
        resolved["output_format"] = getattr(request, "output_format", "png")
        resolved["response_format"] = getattr(request, "response_format", "b64_json")
        return resolved

    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.cfg.model_id,
            "status": "loaded" if self.model_loaded else "not_loaded",
            "error": self._model_error,
            "max_width": self.cfg.max_pixels // self.cfg.dim_alignment,
            "max_height": self.cfg.max_pixels // self.cfg.dim_alignment,
            "max_pixels": self.cfg.max_pixels,
            "default_steps": self.cfg.default_steps,
            "default_guidance": self.cfg.default_guidance,
            "quantization": self.cfg.quantization,
            "backend": "mlx",
        }
