"""Diffusion predictor: abstraction over SD and FLUX inference.

Each predictor wraps a diffusers pipeline and exposes a uniform
``generate()`` interface used by the serve layer. ``warmup()`` is
called at deployment init so the first real request is fast.
"""

from __future__ import annotations

import base64
import gc
import io
import secrets
from abc import ABC, abstractmethod
from typing import Any

import PIL.Image
import torch

from sentimentizer import logger
from sentimentizer.diffusion.config import (
    FLUX2_KLEIN_DEFAULT_CONFIG,
    SD35_DEFAULT_CONFIG,
    SDXL_DEFAULT_CONFIG,
    DiffusionModelConfig,
)

_MAX_SEED = 2**32 - 1
_REF_MAX_PIXELS = 512 * 512


def _decode_b64_image(b64: str, max_pixels: int = _REF_MAX_PIXELS) -> PIL.Image.Image:
    """Decode a base64 image string (raw or data URL) to PIL RGB.

    Raises ValueError on malformed input or images exceeding max_pixels.
    """
    try:
        if b64.startswith("data:image/") and ";base64," in b64:
            b64 = b64.split(";base64,", 1)[1]
        data = base64.b64decode(b64)
        image = PIL.Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"malformed base64 image: {exc}") from exc

    if image.width * image.height > max_pixels:
        raise ValueError(
            f"reference image exceeds max_pixels={max_pixels} ({image.width}x{image.height})"
        )

    return image


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
        reference_images: list[PIL.Image.Image] | None = None,
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


class SDXLPredictor(DiffusionPredictor):
    """SDXL predictor wrapping diffusers StableDiffusionXLPipeline.

    Fits comfortably within 11 GB VRAM (fp16/bfloat16 ~6.5 GB) and supports
    drop-in anime fine-tunes (Illustrious XL, NoobAI XL, etc.) via model_id.
    """

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        super().__init__(cfg or SDXL_DEFAULT_CONFIG)

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from diffusers import StableDiffusionXLPipeline

            dtype = _resolve_dtype(self.cfg.dtype, self._device)
            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "use_safetensors": True,
                "variant": "fp16",
            }
            try:
                self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.cfg.model_id, **load_kwargs
                )
            except (OSError, FileNotFoundError):
                logger.warning(
                    "SDXL fp16 variant not found, retrying without variant",
                    model_id=self.cfg.model_id,
                )
                load_kwargs.pop("variant")
                self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.cfg.model_id, **load_kwargs
                )
            self._pipeline.to(self._device)
            self._model_loaded = True
            logger.info("SDXL model warmed up", model_id=self.cfg.model_id, device=self._device)
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("SDXL warmup failed")

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
        if reference_images is not None:
            raise NotImplementedError("reference_images not supported by SDXL")
        if not self._model_loaded:
            raise RuntimeError(f"SDXL model not loaded: {self._model_error}")

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
            offload = self.cfg.cpu_offload
            if offload == "sequential":
                self._pipeline.enable_sequential_cpu_offload()
            elif offload == "model":
                self._pipeline.enable_model_cpu_offload()
            elif offload is None:
                self._pipeline.to(self._device)
            else:
                raise ValueError(
                    f"Invalid cpu_offload={offload!r}; expected None, 'model', or 'sequential'"
                )
            self._model_loaded = True
            logger.info(
                "SD3.5 model warmed up",
                model_id=self.cfg.model_id,
                device=self._device,
                cpu_offload=offload,
            )
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
        reference_images: list[PIL.Image.Image] | None = None,
    ) -> tuple[Any, int]:
        if reference_images is not None:
            raise NotImplementedError("reference_images not supported by SD3.5")
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
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        result = self._pipeline(**call_kwargs)
        return result.images[0], used_seed


class Flux2KleinPredictor(DiffusionPredictor):
    """FLUX.2 Klein predictor wrapping diffusers Flux2KleinPipeline.

    Klein is the step-distilled, Apache-2.0 variant of FLUX.2 sized for
    consumer GPUs (~13 GB fp16 at native placement). Generation is
    typically 4 steps with guidance_scale=0 — the model is unguided by
    construction.
    """

    def __init__(self, cfg: DiffusionModelConfig | None = None) -> None:
        super().__init__(cfg or FLUX2_KLEIN_DEFAULT_CONFIG)

    def warmup(self) -> None:
        if self._model_loaded:
            return
        try:
            from diffusers import Flux2KleinPipeline

            dtype = _resolve_dtype(self.cfg.dtype, self._device)
            quant = (self.cfg.quantization or "").lower() or None
            pipeline_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if quant is not None:
                pipeline_kwargs.update(self._build_quantized_components(quant, dtype))
            self._pipeline = Flux2KleinPipeline.from_pretrained(
                self.cfg.model_id,
                **pipeline_kwargs,
            )

            offload = self.cfg.cpu_offload
            if offload == "sequential":
                self._pipeline.enable_sequential_cpu_offload()
            elif offload == "model":
                self._pipeline.enable_model_cpu_offload()
            elif offload is None:
                # bitsandbytes places quantized weights itself; .to() would error.
                if quant is None:
                    self._pipeline.to(self._device)
            else:
                raise ValueError(
                    f"Invalid cpu_offload={offload!r}; expected None, 'model', or 'sequential'"
                )

            if quant is not None and hasattr(self._pipeline, "vae"):
                self._pipeline.vae.enable_slicing()
                self._pipeline.vae.enable_tiling()
                self._pipeline.vae.to(self._device)

            self._model_loaded = True
            logger.info(
                "FLUX.2 Klein model warmed up",
                model_id=self.cfg.model_id,
                device=self._device,
                cpu_offload=offload,
                quantization=quant,
            )
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("FLUX.2 Klein warmup failed")

    def _build_quantized_components(self, quant: str, dtype: torch.dtype) -> dict[str, Any]:
        """Build bitsandbytes-quantized transformer + Qwen3 text encoder.

        Allows FLUX.2 Klein to fit on consumer GPUs (~5 GB peak at nf4,
        ~9 GB at int8). Quantized modules place themselves on GPU at
        load time; do not call ``pipeline.to(device)`` afterward.
        """
        try:
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
            from diffusers import Flux2Transformer2DModel
            from transformers import BitsAndBytesConfig as TransformersBnbConfig
            from transformers import Qwen3ForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                f"FLUX.2 Klein quantization={quant!r} requires bitsandbytes "
                f"(install with `uv pip install bitsandbytes`): {exc}"
            ) from exc

        if quant in ("nf4", "int4", "4bit"):
            diffusers_bnb = DiffusersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            transformers_bnb = TransformersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        elif quant in ("int8", "8bit"):
            diffusers_bnb = DiffusersBnbConfig(load_in_8bit=True)
            transformers_bnb = TransformersBnbConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Invalid quantization={quant!r}; expected 'nf4' or 'int8'")

        transformer = Flux2Transformer2DModel.from_pretrained(
            self.cfg.model_id,
            subfolder="transformer",
            quantization_config=diffusers_bnb,
            torch_dtype=dtype,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.cfg.model_id,
            subfolder="text_encoder",
            quantization_config=transformers_bnb,
            torch_dtype=dtype,
        )
        return {"transformer": transformer, "text_encoder": text_encoder}

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
        if not self._model_loaded:
            raise RuntimeError(f"FLUX.2 Klein model not loaded: {self._model_error}")

        used_seed = self._resolve_seed(seed)
        generator = torch.Generator(device=_generator_device(self._device)).manual_seed(used_seed)
        # Klein is unguided (step-distilled); negative_prompt has no effect and
        # isn't accepted by Flux2KleinPipeline. Silently drop instead of erroring
        # so callers can share GenerateRequest across all image models.
        del negative_prompt

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
        if reference_images is not None:
            call_kwargs["image"] = reference_images

        try:
            result = self._pipeline(**call_kwargs)
            return result.images[0], used_seed
        finally:
            gc.collect()
            torch.cuda.empty_cache()
