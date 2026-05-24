"""Diffusion predictor: abstraction over SD and FLUX inference.

Each predictor wraps a diffusers pipeline and exposes a uniform
``generate()`` interface used by the serve layer. ``warmup()`` is
called at deployment init so the first real request is fast.
"""

from __future__ import annotations

import base64
import io
import secrets
from abc import ABC, abstractmethod
from typing import Any

import torch

from sentimentizer import logger
from sentimentizer.diffusion.config import (
    FLUX_DEFAULT_CONFIG,
    SD35_DEFAULT_CONFIG,
    SD_DEFAULT_CONFIG,
    DiffusionModelConfig,
)

_MAX_SEED = 2**32 - 1


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _generator_device(device: str) -> str:
    if device == "mps":
        return "cpu"
    return device


def _resolve_dtype(dtype_str: str, device: str = "cuda") -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    chosen = mapping.get(dtype_str, torch.bfloat16)
    if device == "mps" and chosen == torch.bfloat16:
        return torch.float16
    return chosen


def _check_gguf_device(device: str) -> None:
    if device == "mps":
        raise RuntimeError(
            "GGUF quantization requires CUDA. "
            "Use full-precision weights on MPS by omitting "
            "model_path or pointing to a non-.gguf checkpoint."
        )


def _generate_id() -> str:
    return "img_" + base64.b32encode(secrets.token_bytes(8)).decode("ascii")[:12]


def _encode_pil(image: Any, format: str = "png") -> bytes:
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if format == "webp":
        kwargs["quality"] = 85
    elif format == "jpeg":
        kwargs["quality"] = 90
    image.save(buf, format=format.upper(), **kwargs)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class DiffusionPredictor(ABC):
    """Base class for diffusion model predictors."""

    def __init__(self, cfg: DiffusionModelConfig) -> None:
        self.cfg = cfg
        self._device = _resolve_device()
        self._model_loaded = False
        self._model_error: str | None = None
        self._pipeline: Any = None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_error(self) -> str | None:
        return self._model_error

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> tuple[Any, int]:
        """Generate an image. Returns (PIL.Image, used_seed)."""
        ...

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
        }

    def _resolve_seed(self, seed: int | None) -> int:
        if seed is not None:
            if not (0 <= seed <= _MAX_SEED):
                raise ValueError(f"seed must be 0..{_MAX_SEED}, got {seed}")
            return seed
        return torch.Generator().seed() % (_MAX_SEED + 1)


class SDPredictor(DiffusionPredictor):
    """Stable Diffusion 2.1 predictor wrapping diffusers StableDiffusionPipeline."""

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        super().__init__(cfg or SD_DEFAULT_CONFIG)

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from diffusers import StableDiffusionPipeline

            dtype = _resolve_dtype(self.cfg.dtype, self._device)
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfg.model_id,
                torch_dtype=dtype,
            )
            self._pipeline.to(self._device)
            self._model_loaded = True
            logger.info("SD model warmed up", model_id=self.cfg.model_id, device=self._device)
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("SD warmup failed")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> tuple[Any, int]:
        if not self._model_loaded:
            raise RuntimeError(f"SD model not loaded: {self._model_error}")

        used_seed = self._resolve_seed(seed)
        generator = torch.Generator(device=_generator_device(self._device)).manual_seed(used_seed)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": (steps if steps is not None else self.cfg.default_steps),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None else self.cfg.default_guidance
            ),
            "width": width,
            "height": height,
            "generator": generator,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        result = self._pipeline(**call_kwargs)
        return result.images[0], used_seed


class FluxPredictor(DiffusionPredictor):
    """FLUX.1-dev predictor wrapping diffusers FluxPipeline with GGUF quantization."""

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        super().__init__(cfg or FLUX_DEFAULT_CONFIG)

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from diffusers import FluxPipeline

            dtype = _resolve_dtype(self.cfg.dtype, self._device)
            load_kwargs: dict[str, Any] = {"torch_dtype": dtype}

            if self.cfg.model_path and self.cfg.model_path.endswith(".gguf"):
                _check_gguf_device(self._device)
                from diffusers import GGUFQuantizationConfig

                load_kwargs["quantization_config"] = GGUFQuantizationConfig(
                    compute_dtype=dtype,
                )
                self._pipeline = FluxPipeline.from_single_file(
                    self.cfg.model_path,
                    **load_kwargs,
                )
            else:
                self._pipeline = FluxPipeline.from_pretrained(
                    self.cfg.model_id or self.cfg.model_path,
                    **load_kwargs,
                )

            self._pipeline.to(self._device)
            self._model_loaded = True
            logger.info("FLUX model warmed up", model_id=self.cfg.model_id, device=self._device)
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("FLUX warmup failed")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> tuple[Any, int]:
        if not self._model_loaded:
            raise RuntimeError(f"FLUX model not loaded: {self._model_error}")

        used_seed = self._resolve_seed(seed)
        generator = torch.Generator(device=_generator_device(self._device)).manual_seed(used_seed)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": (steps if steps is not None else self.cfg.default_steps),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None else self.cfg.default_guidance
            ),
            "width": width,
            "height": height,
            "generator": generator,
        }

        result = self._pipeline(**call_kwargs)
        return result.images[0], used_seed


class SD35Predictor(DiffusionPredictor):
    """Stable Diffusion 3.5 Medium predictor wrapping StableDiffusion3Pipeline."""

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        super().__init__(cfg or SD35_DEFAULT_CONFIG)

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from diffusers import StableDiffusion3Pipeline

            dtype = _resolve_dtype(self.cfg.dtype, self._device)
            load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            self._pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.cfg.model_id,
                **load_kwargs,
            )
            self._pipeline.to(self._device)
            self._model_loaded = True
            logger.info("SD3.5 model warmed up", model_id=self.cfg.model_id, device=self._device)
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("SD3.5 warmup failed")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> tuple[Any, int]:
        if not self._model_loaded:
            raise RuntimeError(f"SD3.5 model not loaded: {self._model_error}")

        used_seed = self._resolve_seed(seed)
        generator = torch.Generator(device=_generator_device(self._device)).manual_seed(used_seed)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": (steps if steps is not None else self.cfg.default_steps),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None else self.cfg.default_guidance
            ),
            "width": width,
            "height": height,
            "generator": generator,
            "max_sequence_length": 256,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        result = self._pipeline(**call_kwargs)
        return result.images[0], used_seed
