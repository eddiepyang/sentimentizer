# MLX Diffusion Backend for FLUX.2 Klein

> **Status: COMPLETED** — May 28, 2026

## Background

The diffusion serving pipeline runs three models — SDXL, SD3.5, and FLUX.2 Klein —
all via HuggingFace `diffusers` pipelines backed by PyTorch. On Apple Silicon
Macs, PyTorch uses the MPS backend, which is functional but slow: ~18–20s per
1024×1024 image for FLUX.2 Klein on an M3 Ultra.

The `mflux` library (2.1k stars, MIT license, v0.17.5) is a native MLX port of
FLUX-family diffusion models that achieves **~3.7s steady-state** on the same
hardware — a **4.5–5× speedup** with ~5s cold start (vs ~20s on MPS). It has
zero heavy dependencies (only `mlx`, `tokenizers`, `numpy`; no PyTorch or
`diffusers` at runtime).

However, **mflux only supports FLUX-family models** (FLUX.1, FLUX.2 Klein,
Z-Image, FIBO, Qwen Image, SeedVR2, Depth Pro). There are no mature MLX
implementations of SDXL's U-Net or SD3.5's MMDiT architecture. SDXL and SD3.5
must remain on `diffusers`.

This plan adds an optional `mlx-diffusion` dependency that provides an MLX
backend for **FLUX.2 Klein only**. When `mflux` is installed and the device is
Apple Silicon, `backend="auto"` routes FLUX.2 Klein to MLX automatically.
SDXL and SD3.5 always use `diffusers`. Zero impact on existing deployments.

---

## Decisions Made

| Question | Decision | Rationale |
|---|---|---|
| Scope | FLUX.2 Klein only | mflux has no SDXL or SD3.5 implementation. No mature alternative exists. |
| Default backend | `"auto"` | Zero-config: MLX on Apple Silicon, diffusers otherwise. Explicit `"diffusers"` or `"mlx"` for manual override. |
| `mflux` version | `>=0.17.0` | Minimum version with `Flux2Klein` class and `Flux2KleinEdit` (reference image support). |
| Dependency group | `[mlx-diffusion]` | Separate from `[diffusion]`; installs `mflux` only (which pulls `mlx`, `tokenizers`, `numpy`). |
| Quantization mapping | Config `"nf4"`/`"int4"`/`"4bit"` → mflux `quantize=4`; `"int8"`/`"8bit"` → `quantize=8`; `None` → no quantization | Same semantics as bitsandbytes but works on Apple Silicon. |
| Seed handling | Override `_resolve_seed()` to avoid `torch.Generator` | MLX predictors must not require `torch` for seed generation. Use `os.urandom()` instead. |
| `DiffusionPredictor.__init__` coupling | Override in MLX predictor to skip `_resolve_device()` | Base class calls `torch.backends.mps.is_available()` — MLX doesn't need torch device resolution. |
| `cpu_offload` / `dtype` | MLX predictor ignores these with a warning log | MLX uses unified memory; no CPU offloading. `dtype` is managed by MLX internally. |
| `reference_images` support | Deferred to v2 | mflux's `Flux2KleinEdit` has a different API (`image_paths` not `PIL.Image`). Needs adapter work. |
| `negative_prompt` for FLUX.2 Klein | Silently dropped (matches diffusers behavior) | FLUX.2 Klein is unguided; both backends ignore it. |
| `model_info()` | Add `"backend"` field | Monitoring needs to know which backend served the request. |

---

## Out of Scope

- **SDXL and SD3.5 MLX predictors**: No library provides these. When they appear, add `MLX*Predictor` classes and update `BACKEND_REGISTRY`.
- **`reference_images` for MLX FLUX.2 Klein**: Requires bridging `PIL.Image` → `image_paths` and using `Flux2KleinEdit` instead of `Flux2Klein`. Separate plan.
- **Pre-quantized model loading**: Loading community models like `mlx-community/flux2-klein-4b-8bit` directly via `model_path`. Post-v1.
- **FLUX.1, Z-Image, FIBO, Qwen Image**: Not in the current config. Add when needed.
- **Training/fine-tuning via mflux**: mflux has LoRA training support but this plan covers inference only.

---

## Architecture

### Backend resolution

```
backend="auto" on Apple Silicon with mflux installed
  └─→ "mlx"   (for FLUX.2 Klein only)
  └─→ "diffusers" (for SDXL, SD3.5)

backend="auto" on CUDA / CPU / no mflux
  └─→ "diffusers" (all models)
```

### File tree

```
sentimentizer/diffusion/
  __init__.py              # Add: conditional MLXFlux2KleinPredictor export
  config.py                # Modify: add backend field, BACKEND_REGISTRY, resolve_backend()
  diffusion_config.yaml    # Modify: add backend: "auto" / "diffusers" entries
  predictor.py             # Modify: add create_predictor() factory, _PREDICTOR_REGISTRY
  mlx_compat.py            # NEW: MFLUX_AVAILABLE, is_mlx_device()
  mlx_predictor.py         # NEW: MLXFlux2KleinPredictor
  job_store.py             # Unchanged
```

### Factory pattern

`create_predictor(model_key, cfg)` replaces direct predictor instantiation in
`diffusion_app.py`. Callers don't know or care which backend serves the request:

```python
# Before:
self.predictor = Flux2KleinPredictor(model_cfg)

# After:
self.predictor = create_predictor("flux2_klein", model_cfg)
# Returns MLXFlux2KleinPredictor on Apple Silicon with mflux,
# Flux2KleinPredictor otherwise.
```

### Import guard

Following the existing `TRANSFORMERS_AVAILABLE` pattern in `hf_base.py`:

```python
# mlx_compat.py
try:
    import mflux  # noqa: F401
    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False

def is_mlx_device() -> bool:
    try:
        import mlx.core as mx
        # mx.default_device() returns a Device with a .type attribute.
        return mx.default_device().type == mx.DeviceType.gpu
    except ImportError:
        return False
```

`MLXFlux2KleinPredictor` is only imported when `MFLUX_AVAILABLE` is `True`.
The import guard is in `predictor.py` and `__init__.py`, so a pure-diffusers
environment never touches `mflux` or `mlx`.

---

## Detailed File Changes

### 1. `pyproject.toml` — Add optional dependency group

```toml
[project.optional-dependencies]
# ... existing groups ...
mlx-diffusion = [
    "mflux>=0.17.0",
]
```

Notes:
- `mlx` is a transitive dependency of `mflux`; no need to list it separately.
- `mflux` depends only on `mlx`, `tokenizers`, `numpy`, and `huggingface_hub` — no conflict with `torch` or `diffusers`.

### 2. `sentimentizer/diffusion/mlx_compat.py` (NEW)

```python
"""MLX availability detection — no module-level torch import."""

from __future__ import annotations

try:
    import mflux  # noqa: F401

    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False


def is_mlx_device() -> bool:
    """True when running on Apple Silicon with MLX GPU available."""
    try:
        import mlx.core as mx

        return mx.default_device() == mx.Device.gpu
    except (ImportError, Exception):
        return False
```

### 3. `sentimentizer/diffusion/config.py` — Add `backend` field and registry

Add to imports:

```python
from typing import Literal
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE, is_mlx_device
```

Add `backend` field to `DiffusionModelConfig`:

```python
@dataclass(frozen=True)
class DiffusionModelConfig:
    # ... existing fields ...
    backend: Literal["diffusers", "mlx", "auto"] = "auto"
```

Add registry and resolver after the class:

```python
BACKEND_REGISTRY: dict[str, list[str]] = {
    "sdxl": ["diffusers"],
    "sd35": ["diffusers"],
    "flux2_klein": ["diffusers"] + (["mlx"] if MFLUX_AVAILABLE else []),
}


def resolve_backend(model_key: str, backend: str = "auto") -> str:
    """Resolve 'auto' to a concrete backend based on model and device.

    For SDXL and SD3.5, always returns 'diffusers' (no MLX implementation).
    For FLUX.2 Klein, returns 'mlx' on Apple Silicon when mflux is installed,
    otherwise 'diffusers'.
    """
    available = BACKEND_REGISTRY.get(model_key, ["diffusers"])
    if backend != "auto":
        if backend not in available:
            raise ValueError(
                f"backend={backend!r} not available for {model_key}. "
                f"Available: {available}"
            )
        return backend
    if "mlx" in available and is_mlx_device():
        return "mlx"
    return "diffusers"
```

Do **not** add a `backend` entry to `_ENV_OVERRIDES` in
`sentimentizer/diffusion/config.py`. The existing convention is that
`sentimentizer/diffusion/config.py` env vars own *model-internal* defaults
(`default_steps`, `default_guidance`, `max_pixels`), while `ServeConfig`
env vars own *operational* overrides (`model_id`, `cpu_offload`,
`quantization`). Backend selection is operational, so the env var
`SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND` lives in `ServeConfig` only
(see step 8). The dispatcher applies it on top of the YAML default via
`dataclasses.replace()`, matching the existing `model_id` flow.

### 4. `sentimentizer/diffusion/diffusion_config.yaml` — Add `backend` entries

```yaml
sd35:
  # ... existing fields ...
  backend: "diffusers"   # No MLX implementation exists for SD3.5

sdxl:
  # ... existing fields ...
  backend: "diffusers"   # No MLX implementation exists for SDXL

flux2_klein:
  # ... existing fields ...
  backend: "auto"        # "auto" uses MLX on Apple Silicon, diffusers otherwise
```

Hardcoding `sd35` and `sdxl` to `"diffusers"` prevents user confusion —
no one can accidentally set `backend="mlx"` for a model that doesn't support it.

### 5. `sentimentizer/diffusion/mlx_predictor.py` (NEW)

```python
"""MLX-based diffusion predictors for Apple Silicon."""

from __future__ import annotations

import logging
from typing import Any

import PIL.Image

from sentimentizer import logger
from sentimentizer.diffusion.config import FLUX2_KLEIN_DEFAULT_CONFIG, DiffusionModelConfig
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE

_MFLUX_QUANTIZE_MAP: dict[str | None, int | None] = {
    None: None,
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
        self._backend_name: str = "mlx"

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
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
            from mflux.models.common.config import ModelConfig

            quantize = _MFLUX_QUANTIZE_MAP.get(
                (self.cfg.quantization or "").lower() or None
            )
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
        if not self._model_loaded:
            raise RuntimeError(
                f"MLX FLUX.2 Klein model not loaded: {self._model_error}"
            )

        used_seed = self._resolve_seed_mlx(seed)

        # FLUX.2 Klein is unguided; negative_prompt is silently dropped.
        # (Same behavior as the diffusers Flux2KleinPredictor.)
        del negative_prompt

        # reference_images not yet supported for MLX backend (v2).
        # Flux2KleinEdit has a different API (image_paths, not PIL.Image).
        if reference_images is not None:
            raise NotImplementedError(
                "reference_images are not yet supported by the MLX backend. "
                "Use backend='diffusers' or set SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND=diffusers."
            )

        try:
            result = self._pipeline.generate_image(
                seed=used_seed,
                prompt=prompt,
                num_inference_steps=(
                    steps if steps is not None else self.cfg.default_steps
                ),
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
        resolved["steps"] = (
            request.steps if request.steps is not None else self.cfg.default_steps
        )
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
            "backend": self._backend_name,
        }
```

Key design notes:
- **Does NOT extend `DiffusionPredictor`**: The base class `__init__` calls
  `_resolve_device()` and `__init__` creates `torch.Generator` in
  `_resolve_seed()`. Instead, we replicate the interface (`warmup()`,
  `generate()`, `model_info()`, `resolve_defaults()`) as a standalone class.
  This avoids any `torch` dependency in the MLX path.
- **`_resolve_seed_mlx()`** uses `os.urandom()` instead of `torch.Generator().seed()`.
- **`reference_images` raises `NotImplementedError`**: mflux's `Flux2KleinEdit`
  uses `image_paths` (file paths), not `PIL.Image` objects. v2 will bridge this.
- **`model_info()` includes `"backend": "mlx"`**: Monitoring can distinguish backends.

### 5b. `sentimentizer/diffusion/predictor.py` — Emit `backend` from base `model_info()`

For monitoring symmetry, the base `DiffusionPredictor.model_info()` must
also return `"backend": "diffusers"` so the field is always present
regardless of backend. Without this, dashboards filtering on `backend`
silently drop diffusers-served requests.

```python
# DiffusionPredictor.model_info() — add one line:
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
        "backend": "diffusers",   # NEW
    }
```

The MLX predictor already sets `"backend": "mlx"` in its own
`model_info()`, so step 10b's `_get_backend()` cache works for both
backend families.

### 6. `sentimentizer/diffusion/predictor.py` — Add Protocol, factory, and registry

The MLX predictor deliberately doesn't inherit from `DiffusionPredictor`
(the base `__init__` calls `_resolve_device()` which imports torch). To
keep the factory typed without lying about the return type, define a
`Protocol` that both predictor families satisfy structurally:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DiffusionPredictorProtocol(Protocol):
    """Structural interface for diffusion predictors (torch or MLX).

    Both DiffusionPredictor subclasses and MLXFlux2KleinPredictor satisfy
    this protocol. The factory returns this type so callers get accurate
    autocomplete and type checking regardless of backend.
    """

    @property
    def model_loaded(self) -> bool: ...
    @property
    def model_error(self) -> str | None: ...

    def warmup(self) -> None: ...
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
    ) -> tuple[Any, int]: ...
    def resolve_defaults(self, request: Any) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, Any]: ...
```

`DiffusionPredictor` (existing ABC) already satisfies this protocol
structurally; no inheritance change required. `MLXFlux2KleinPredictor`
satisfies it because it implements the same methods.

Add registry and factory at module level:

```python
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE

_PREDICTOR_REGISTRY: dict[str, dict[str, type]] = {
    "sdxl": {"diffusers": SDXLPredictor},
    "sd35": {"diffusers": SD35Predictor},
    "flux2_klein": {"diffusers": Flux2KleinPredictor},
}

if MFLUX_AVAILABLE:
    from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

    _PREDICTOR_REGISTRY["flux2_klein"]["mlx"] = MLXFlux2KleinPredictor
```

Add factory function:

```python
def create_predictor(
    model_key: str,
    cfg: DiffusionModelConfig | None = None,
) -> DiffusionPredictorProtocol:
    """Create the appropriate predictor based on config backend and availability.

    Args:
        model_key: One of "sdxl", "sd35", "flux2_klein".
        cfg: Optional DiffusionModelConfig. Uses model defaults if None.

    Returns:
        A predictor instance satisfying DiffusionPredictorProtocol.
        Concretely: a DiffusionPredictor subclass for backend="diffusers",
        or MLXFlux2KleinPredictor for backend="mlx".

    Raises:
        ValueError: If the requested backend is not available for the model.
        ImportError: If MLX backend is requested but mflux is not installed.
    """
    from sentimentizer.diffusion.config import resolve_backend

    if cfg is None:
        defaults = {
            "sdxl": SDXL_DEFAULT_CONFIG,
            "sd35": SD35_DEFAULT_CONFIG,
            "flux2_klein": FLUX2_KLEIN_DEFAULT_CONFIG,
        }
        cfg = defaults[model_key]

    backend = resolve_backend(model_key, cfg.backend)
    available = _PREDICTOR_REGISTRY.get(model_key, {})

    if backend not in available:
        raise ValueError(
            f"backend={backend!r} not available for {model_key}. "
            f"Available: {list(available.keys())}"
        )

    predictor_cls = available[backend]

    if backend == "mlx":
        if cfg.cpu_offload:
            logger.warning(
                "cpu_offload=%r is not supported by the MLX backend; "
                "MLX uses unified memory. Ignoring.",
                cfg.cpu_offload,
            )
        if cfg.dtype not in ("bfloat16", "float16", ""):
            logger.warning(
                "dtype=%r — MLX manages precision internally. Ignoring.",
                cfg.dtype,
            )

    return predictor_cls(cfg)
```

### 7. `sentimentizer/diffusion/__init__.py` — Conditional exports

```python
from sentimentizer.diffusion.config import DiffusionModelConfig
from sentimentizer.diffusion.job_store import JobStore
from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE
from sentimentizer.diffusion.predictor import (
    DiffusionPredictor,
    DiffusionPredictorProtocol,
    Flux2KleinPredictor,
    SD35Predictor,
    SDXLPredictor,
    create_predictor,
)

__all__ = [
    "DiffusionModelConfig",
    "DiffusionPredictor",
    "DiffusionPredictorProtocol",
    "Flux2KleinPredictor",
    "JobStore",
    "SD35Predictor",
    "SDXLPredictor",
    "create_predictor",
]

if MFLUX_AVAILABLE:
    from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

    __all__.append("MLXFlux2KleinPredictor")
```

### 8. `sentimentizer/serve/config.py` — Add backend env var

Add field to `ServeConfig`:

```python
flux2_klein_backend: str = "auto"   # "auto", "diffusers", or "mlx"
```

Add to `_ENV_OVERRIDES`:

```python
"SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND": "flux2_klein_backend",
```

### 9. `sentimentizer/serve/service.yaml` — Add backend config

```yaml
flux2_klein_backend: "auto"    # "auto", "diffusers", or "mlx"
```

### 10. `sentimentizer/serve/diffusion_app.py` — Use factory + thread backend override

For `Flux2KleinDeployment.__init__`, extend the existing override block
to also apply `cfg.flux2_klein_backend`, then swap construction to the factory:

```python
# Before (Flux2KleinDeployment.__init__):
overrides: dict[str, Any] = {}
if cfg.flux2_klein_model_id:
    overrides["model_id"] = cfg.flux2_klein_model_id
if cfg.flux2_klein_cpu_offload:
    overrides["cpu_offload"] = cfg.flux2_klein_cpu_offload
if cfg.flux2_klein_quantization:
    overrides["quantization"] = cfg.flux2_klein_quantization
model_cfg = (
    replace(FLUX2_KLEIN_DEFAULT_CONFIG, **overrides)
    if overrides
    else FLUX2_KLEIN_DEFAULT_CONFIG
)
self.predictor = Flux2KleinPredictor(model_cfg)

# After:
from sentimentizer.diffusion.predictor import create_predictor

overrides: dict[str, Any] = {}
if cfg.flux2_klein_model_id:
    overrides["model_id"] = cfg.flux2_klein_model_id
if cfg.flux2_klein_cpu_offload:
    overrides["cpu_offload"] = cfg.flux2_klein_cpu_offload
if cfg.flux2_klein_quantization:
    overrides["quantization"] = cfg.flux2_klein_quantization
if cfg.flux2_klein_backend:                  # NEW: thread backend override
    overrides["backend"] = cfg.flux2_klein_backend
model_cfg = (
    replace(FLUX2_KLEIN_DEFAULT_CONFIG, **overrides)
    if overrides
    else FLUX2_KLEIN_DEFAULT_CONFIG
)
self.predictor = create_predictor("flux2_klein", model_cfg)
self._backend = model_cfg.backend  # cached for dispatcher introspection
```

Without the new `cfg.flux2_klein_backend` line, the YAML `backend: "auto"`
always wins and the ServeConfig env var is dead.

**Cold-start note**: Bump `health_check_timeout_s` on `Flux2KleinDeployment`
to at least 600s. First-ever warmup downloads 4–13 GB of weights from
HuggingFace; the default 60s health-check timeout would mark the deployment
unhealthy and trigger a restart loop. The MLX path's "~5s cold start" claim
applies only after weights are cached locally. For production, pre-pull
weights into the HuggingFace cache during image build.

For `SD35Deployment.__init__` and `SDXLDeployment.__init__`, just swap
construction (no backend override since `BACKEND_REGISTRY` doesn't offer
"mlx" for those models):

```python
# SD35Deployment
self.predictor = create_predictor("sd35", model_cfg)

# SDXLDeployment — model_id still passed via constructor arg + replace()
self.predictor = create_predictor("sdxl", model_cfg)
```

Also expose the resolved backend on `info()` so the dispatcher can apply
backend-aware request validation (see step 10b):

```python
def info(self) -> dict[str, Any]:
    return self.predictor.model_info()  # already includes "backend" field
```

### 10b. `sentimentizer/serve/diffusion_app.py` — Backend-aware dispatcher guards

The MLX backend doesn't support `reference_images`. Reject at the dispatcher
(returns 400) instead of letting the request reach the actor and raise 500.

Add a small cache on `ImagesDispatcher`:

```python
def __init__(self, ...) -> None:
    ...
    self._backend_by_model: dict[str, str] = {}  # populated lazily

async def _get_backend(self, model: str) -> str:
    if model not in self._backend_by_model:
        info = await self._handles[model].info.remote()
        self._backend_by_model[model] = info.get("backend", "diffusers")
    return self._backend_by_model[model]
```

Update both `generate_image` and `create_job`. After resolving `model_name`
and validating it's enabled, replace the existing reference_images guard:

```python
# Before:
if body.reference_images is not None:
    if model_name != "flux2_klein":
        raise HTTPException(400, {"code": "reference_images_unsupported", ...})

# After:
if body.reference_images is not None:
    if model_name != "flux2_klein":
        raise HTTPException(400, {"code": "reference_images_unsupported",
                                  "message": "Reference images are only supported by FLUX.2 Klein"})
    backend = await self._get_backend(model_name)
    if backend == "mlx":
        raise HTTPException(
            400,
            {"code": "reference_images_unsupported_backend",
             "message": "reference_images require backend='diffusers'; "
                        "current FLUX.2 Klein backend is 'mlx'. "
                        "Set SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND=diffusers."},
        )
```

### 11. Tests — `tests/test_mlx_predictor.py` (NEW)

```python
"""Tests for MLX diffusion backend (no mflux required for most tests)."""

import importlib.util
from unittest.mock import patch

import pytest

from sentimentizer.diffusion.config import (
    BACKEND_REGISTRY,
    DiffusionModelConfig,
    resolve_backend,
)
from sentimentizer.diffusion.predictor import (
    Flux2KleinPredictor,
    SD35Predictor,
    SDXLPredictor,
    create_predictor,
)

# Boolean flag — do NOT use `pytest.importorskip` at module scope; that
# would skip the entire file (including the registry/factory tests, which
# don't need mflux at all).
_MFLUX_INSTALLED = importlib.util.find_spec("mflux") is not None


class TestBackendRegistry:
    def test_sdxl_only_has_diffusers(self) -> None:
        assert BACKEND_REGISTRY["sdxl"] == ["diffusers"]

    def test_sd35_only_has_diffusers(self) -> None:
        assert BACKEND_REGISTRY["sd35"] == ["diffusers"]

    def test_flux2_klein_always_has_diffusers(self) -> None:
        assert "diffusers" in BACKEND_REGISTRY["flux2_klein"]


class TestResolveBackend:
    def test_explicit_diffusers_always_works(self) -> None:
        assert resolve_backend("sdxl", "diffusers") == "diffusers"

    def test_mlx_for_sdxl_raises(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            resolve_backend("sdxl", "mlx")

    def test_mlx_for_sd35_raises(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            resolve_backend("sd35", "mlx")

    def test_auto_for_sdxl_always_returns_diffusers(self) -> None:
        assert resolve_backend("sdxl", "auto") == "diffusers"

    def test_auto_for_sd35_always_returns_diffusers(self) -> None:
        assert resolve_backend("sd35", "auto") == "diffusers"

    def test_auto_on_mlx_device_with_mflux_returns_mlx(self) -> None:
        with patch("sentimentizer.diffusion.config.is_mlx_device", return_value=True):
            with patch.dict(BACKEND_REGISTRY, {"flux2_klein": ["diffusers", "mlx"]}):
                result = resolve_backend("flux2_klein", "auto")
                assert result == "mlx"

    def test_auto_on_non_mlx_device_returns_diffusers(self) -> None:
        with patch("sentimentizer.diffusion.config.is_mlx_device", return_value=False):
            result = resolve_backend("flux2_klein", "auto")
            assert result == "diffusers"


class TestCreatePredictor:
    def test_sdxl_returns_sdxl_predictor(self) -> None:
        p = create_predictor("sdxl")
        assert isinstance(p, SDXLPredictor)

    def test_sd35_returns_sd35_predictor(self) -> None:
        p = create_predictor("sd35")
        assert isinstance(p, SD35Predictor)

    def test_flux2_klein_default_returns_diffusers_on_non_mlx(self) -> None:
        with patch("sentimentizer.diffusion.config.is_mlx_device", return_value=False):
            p = create_predictor("flux2_klein")
            assert isinstance(p, Flux2KleinPredictor)

    def test_explicit_backend_diffusers(self) -> None:
        cfg = DiffusionModelConfig(backend="diffusers")
        p = create_predictor("flux2_klein", cfg)
        assert isinstance(p, Flux2KleinPredictor)


class TestDiffusionModelConfigBackendField:
    def test_default_backend_is_auto(self) -> None:
        cfg = DiffusionModelConfig()
        assert cfg.backend == "auto"

    def test_explicit_diffusers(self) -> None:
        cfg = DiffusionModelConfig(backend="diffusers")
        assert cfg.backend == "diffusers"

    def test_explicit_mlx(self) -> None:
        cfg = DiffusionModelConfig(backend="mlx")
        assert cfg.backend == "mlx"

    def test_frozen(self) -> None:
        cfg = DiffusionModelConfig()
        with pytest.raises(AttributeError):
            cfg.backend = "mlx"  # type: ignore[misc]


@pytest.mark.skipif(not _MFLUX_INSTALLED, reason="mflux not installed")
class TestMLXFlux2KleinPredictor:
    def test_init_not_loaded(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_backend_name(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        info = p.model_info()
        assert info["backend"] == "mlx"

    def test_resolve_seed_mlx_deterministic(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        assert p._resolve_seed_mlx(42) == 42

    def test_resolve_seed_mlx_random(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        s = p._resolve_seed_mlx(None)
        assert 0 <= s <= 2**32 - 1

    def test_resolve_seed_mlx_invalid(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        with pytest.raises(ValueError):
            p._resolve_seed_mlx(-1)

    def test_generate_raises_if_not_loaded(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")

    def test_reference_images_raises_not_implemented(self) -> None:
        from PIL import Image
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        p = MLXFlux2KleinPredictor()
        with pytest.raises(NotImplementedError, match="reference_images"):
            p.generate("test prompt", reference_images=[Image.new("RGB", (64, 64))])

    def test_quantize_mapping(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import _MFLUX_QUANTIZE_MAP

        assert _MFLUX_QUANTIZE_MAP["nf4"] == 4
        assert _MFLUX_QUANTIZE_MAP["int4"] == 4
        assert _MFLUX_QUANTIZE_MAP["4bit"] == 4
        assert _MFLUX_QUANTIZE_MAP["int8"] == 8
        assert _MFLUX_QUANTIZE_MAP["8bit"] == 8
        assert _MFLUX_QUANTIZE_MAP[None] is None
```

### 12. `AGENTS.md` — Add convention section

Add under "Important Conventions":

```markdown
### MLX Diffusion Backend

- **MLX is optional**: `pip install sentimentizer[mlx-diffusion]` adds `mflux` for
  Apple Silicon acceleration. Without it, all diffusion models use `diffusers` (PyTorch).
- **`backend` config**: `DiffusionModelConfig.backend` controls which inference
  backend to use: `"auto"` (MLX on Apple Silicon, diffusers otherwise), `"diffusers"`
  (always PyTorch), or `"mlx"` (always MLX, raises ImportError if mflux not installed).
- **Only FLUX.2 Klein has an MLX backend**: SDXL and SD3.5 have no MLX
  implementation. `BACKEND_REGISTRY` defines available backends per model.
  Setting `backend="mlx"` for sd35 or sdxl raises `ValueError`.
- **`MLXFlux2KleinPredictor` does NOT extend `DiffusionPredictor`**: The base class
  calls `torch.backends.mps.is_available()` in `__init__` and uses `torch.Generator`
  for seed resolution. The MLX predictor replicates the interface as a standalone class
  to avoid any `torch` dependency in the MLX path. Both predictor families satisfy
  the `DiffusionPredictorProtocol` (structural typing), so `create_predictor()` is
  typed against the protocol — callers get accurate autocomplete and `mypy`
  catches interface drift regardless of backend.
- **`model_info()` always includes `"backend"` field**: Both the base
  `DiffusionPredictor.model_info()` (`"diffusers"`) and `MLXFlux2KleinPredictor.model_info()`
  (`"mlx"`) emit this key so dashboards can filter on backend without dropping requests.
- **`cpu_offload` and `dtype` are diffusers-only**: MLX predictor ignores these config
  fields and logs warnings if they're set.
- **Quantization mapping**: Config `"nf4"`/`"int4"`/`"4bit"` → mflux `quantize=4`;
  `"int8"`/`"8bit"` → `quantize=8`; `None` → no quantization. No bitsandbytes needed.
- **`reference_images` not yet supported for MLX backend**: Raises
  `NotImplementedError`. Use `backend="diffusers"` for reference image support.
```

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **No MLX for SDXL/SD3.5** | Certain (already confirmed) | Scope reduction | `BACKEND_REGISTRY` only offers `"mlx"` for `flux2_klein`. SDXL/SD3.5 always use diffusers. Adding MLX for those models requires new libraries or ports. |
| **`torch` + `mlx` GPU memory contention** | Low (separate processes in Ray Serve) | Medium (OOM if both loaded in same process) | Log warning if both are loaded. In practice, sentiment training (PyTorch) and diffusion serving (MLX) run in separate Ray deployments. |
| **`mflux` API changes** | Medium (v0.17, actively developed) | Medium (breakage on upgrade) | Pin `mflux>=0.17.0` and test against specific versions. The API surface we use (`Flux2Klein`, `ModelConfig`, `generate_image`) is stable. |
| **`DiffusionPredictor` base class coupling** | Certain (already known) | Low (we work around it) | `MLXFlux2KleinPredictor` replicates the interface without inheriting. If the interface drifts, tests catch it. |
| **`reference_images` API mismatch** | Certain (known gap) | Low (error message is clear) | `NotImplementedError` with helpful message. v2 will bridge `PIL.Image` → `image_paths`. |
| **Quantization config semantic differences** | Low | Low (functionally similar) | On-the-fly mflux quantization (`quantize=4/8`) is equivalent in result to bitsandbytes dynamic quantization. Memory profiles differ slightly but both reduce VRAM usage. |
| **First-run cold start exceeds Ray Serve health-check timeout** | Medium (only on fresh hosts) | High (deployment marked unhealthy → restart loop) | "~5s cold start" is steady-state; first-ever run downloads 4–13 GB of weights. Bump `Flux2KleinDeployment` health-check timeout (`health_check_timeout_s`) to ≥600s, or pre-pull weights into the HuggingFace cache during image build. Document this in the deploy runbook. |

---

## Implementation Order

1. `mlx_compat.py` — Import guard (no other code depends on it yet)
2. `config.py` — `backend` field, `BACKEND_REGISTRY`, `resolve_backend()`
3. `diffusion_config.yaml` — `backend` entries per model
4. `predictor.py` — Add `"backend": "diffusers"` to base `model_info()` (step 5b)
5. `mlx_predictor.py` — `MLXFlux2KleinPredictor` class
6. `predictor.py` — `DiffusionPredictorProtocol`, `create_predictor()` factory, `_PREDICTOR_REGISTRY`
7. `__init__.py` — Conditional `MLXFlux2KleinPredictor` export + `DiffusionPredictorProtocol`
8. `serve/config.py` + `service.yaml` — `flux2_klein_backend` env var
9. `serve/diffusion_app.py` — Replace predictor instantiation with `create_predictor()`, thread `backend` override (step 10)
10. `serve/diffusion_app.py` — Backend-aware dispatcher guard for `reference_images` (step 10b)
11. `pyproject.toml` — `mlx-diffusion` optional dep
12. `tests/test_mlx_predictor.py` — Backend registry, resolve, factory, config, MLX predictor
13. `AGENTS.md` — Convention docs

Each step is independently testable: `make check` (lint + existing tests) should pass after each step.