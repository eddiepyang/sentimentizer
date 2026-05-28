"""Diffusion serving: dispatcher, SD3.5/FLUX.2 Klein/SDXL deployments, and image routes.

Uses the Ray Serve composition pattern: a lightweight CPU dispatcher
holds DeploymentHandles to GPU-backed model deployments. Routes are
attached to the dispatcher via ``@app.post`` / ``@app.get`` on the same
FastAPI app from ``sentimentizer/serve/app.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import orjson
import torch
from fastapi import Depends, HTTPException, Query, Request, Response

from sentimentizer import logger
from sentimentizer.diffusion.config import (
    FLUX2_KLEIN_DEFAULT_CONFIG,
    SD35_DEFAULT_CONFIG,
    SDXL_DEFAULT_CONFIG,
)
from sentimentizer.diffusion.image_utils import (
    _REF_MAX_PIXELS,
    b64_encode,
    decode_b64_image,
    encode_pil,
    generate_id,
)
from sentimentizer.diffusion.predictor import create_predictor
from sentimentizer.serve.app import create_fastapi_app
from sentimentizer.serve.base import ServiceMetrics, serve
from sentimentizer.serve.config import cfg
from sentimentizer.serve.diffusion_models import (
    GenerateRequest,
    GenerateResponse,
    ImageModelDetailResponse,
    ImageModelsResponse,
    JobListResponse,
    JobResponse,
)
from sentimentizer.serve.middleware import (
    IdempotencyCache,
    check_prompt_safety,
    idempotent,
    rate_limit,
    require_api_key,
)


def _body_hash(body: GenerateRequest) -> str:
    raw = orjson.dumps(body.model_dump(mode="python"), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:32]


_num_gpus = 1 if torch.cuda.is_available() else 0

_PREDICTOR_DEFAULTS: dict[str, Any] = {
    "flux2_klein": FLUX2_KLEIN_DEFAULT_CONFIG,
    "sd35": SD35_DEFAULT_CONFIG,
    "sdxl": SDXL_DEFAULT_CONFIG,
}


def _validate_reference_images(
    body: GenerateRequest,
    model_name: str,
) -> list[Any] | None:
    """Decode and validate reference_images from the request body.

    Returns the decoded PIL image list, or None if no reference images.
    Raises HTTPException on validation failure.
    """
    if body.reference_images is None:
        return None
    if model_name != "flux2_klein":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "reference_images_unsupported",
                "message": "Reference images are only supported by FLUX.2 Klein",
            },
        )
    try:
        return [decode_b64_image(b64, _REF_MAX_PIXELS) for b64 in body.reference_images]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_reference_image",
                "message": str(exc),
            },
        ) from exc


@dataclass
class _PreparedRequest:
    model_name: str
    handle: Any
    reference_images: list[Any] | None
    steps: int
    guidance_scale: float
    max_pixels: int
    dim_alignment: int


images_app = create_fastapi_app(
    title="Sentimentizer Images",
    description="Image generation API (FLUX.2 Klein / SD 3.5 Medium / SDXL)",
    path_limits={"/": 4 * 1024 * 1024},
)


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 2, "num_gpus": _num_gpus},
    health_check_timeout_s=600,
)
class Flux2KleinDeployment:
    def __init__(self) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import FLUX2_KLEIN_DEFAULT_CONFIG

        overrides: dict[str, Any] = {}
        if cfg.flux2_klein_model_id:
            overrides["model_id"] = cfg.flux2_klein_model_id
        if cfg.flux2_klein_cpu_offload:
            overrides["cpu_offload"] = cfg.flux2_klein_cpu_offload
        if cfg.flux2_klein_quantization:
            overrides["quantization"] = cfg.flux2_klein_quantization
        if cfg.flux2_klein_backend:
            overrides["backend"] = cfg.flux2_klein_backend
        model_cfg = (
            replace(FLUX2_KLEIN_DEFAULT_CONFIG, **overrides)
            if overrides
            else FLUX2_KLEIN_DEFAULT_CONFIG
        )
        self.predictor = create_predictor("flux2_klein", model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="flux2_klein")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        start = time.perf_counter()
        try:
            res = await asyncio.to_thread(self.predictor.generate, **kwargs)
        except Exception:
            latency = time.perf_counter() - start
            self._metrics.record_request(latency, error=True)
            raise
        latency = time.perf_counter() - start
        self._metrics.record_request(latency)
        return res

    def info(self) -> dict[str, Any]:
        return self.predictor.model_info()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 2, "num_gpus": _num_gpus},
    health_check_timeout_s=600,
)
class SD35Deployment:
    def __init__(self) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import SD35_DEFAULT_CONFIG

        overrides: dict[str, Any] = {}
        if cfg.sd35_model_id:
            overrides["model_id"] = cfg.sd35_model_id
        if cfg.sd35_cpu_offload:
            overrides["cpu_offload"] = cfg.sd35_cpu_offload
        model_cfg = replace(SD35_DEFAULT_CONFIG, **overrides) if overrides else SD35_DEFAULT_CONFIG
        self.predictor = create_predictor("sd35", model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="sd35")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        start = time.perf_counter()
        try:
            res = await asyncio.to_thread(self.predictor.generate, **kwargs)
        except Exception:
            latency = time.perf_counter() - start
            self._metrics.record_request(latency, error=True)
            raise
        latency = time.perf_counter() - start
        self._metrics.record_request(latency)
        return res

    def info(self) -> dict[str, Any]:
        return self.predictor.model_info()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 2, "num_gpus": _num_gpus},
    health_check_timeout_s=600,
)
class SDXLDeployment:
    def __init__(self, model_id: str) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import SDXL_DEFAULT_CONFIG

        model_cfg = (
            replace(SDXL_DEFAULT_CONFIG, model_id=model_id) if model_id else SDXL_DEFAULT_CONFIG
        )
        self.predictor = create_predictor("sdxl", model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="sdxl")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        start = time.perf_counter()
        try:
            res = await asyncio.to_thread(self.predictor.generate, **kwargs)
        except Exception:
            latency = time.perf_counter() - start
            self._metrics.record_request(latency, error=True)
            raise
        latency = time.perf_counter() - start
        self._metrics.record_request(latency)
        return res

    def info(self) -> dict[str, Any]:
        return self.predictor.model_info()


@serve.deployment(
    num_replicas=2,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(images_app)
class ImagesDispatcher:
    """Front-door deployment with HTTP routes; forwards work to SD/FLUX actors."""

    def __init__(
        self,
        flux2_klein: Any = None,
        sd35: Any = None,
        sdxl: dict[str, Any] | None = None,
    ) -> None:
        self._handles: dict[str, Any] = {}
        if flux2_klein is not None:
            self._handles["flux2_klein"] = flux2_klein
        if sd35 is not None:
            self._handles["sd35"] = sd35
        for name, handle in (sdxl or {}).items():
            self._handles[name] = handle
        self._sdxl_names: set[str] = set(sdxl.keys()) if sdxl else set()
        self._idem = IdempotencyCache(ttl_s=cfg.idempotency_ttl_s)
        self._store: Any = None
        self._refs: dict[str, Any] = {}
        self._poll_tasks: set[asyncio.Task] = set()
        self._backend_by_model: dict[str, str] = {}

    def _get_handle(self, model: str) -> Any:
        if model not in self._handles:
            avail = list(self._handles.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": (f"Model '{model}' is not enabled. " f"Available: {avail}"),
                },
            )
        return self._handles[model]

    async def _get_backend(self, model: str) -> str:
        if model not in self._backend_by_model:
            info = await self._handles[model].info.remote()
            self._backend_by_model[model] = info.get("backend", "diffusers")
        return self._backend_by_model[model]

    def _get_store(self) -> Any:
        import ray

        if self._store is None:
            try:
                self._store = ray.get_actor("diffusion_job_store")
            except ValueError:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "job_store_unavailable",
                        "message": "Job store not initialized",
                    },
                ) from None
        return self._store

    def _get_predictor_defaults(self, model: str) -> dict[str, Any]:
        model_cfg = _PREDICTOR_DEFAULTS.get(model)
        if model_cfg is None and model in self._sdxl_names:
            model_cfg = SDXL_DEFAULT_CONFIG
        if model_cfg is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": f"No default config for model '{model}'",
                },
            )
        return {
            "default_steps": model_cfg.default_steps,
            "default_guidance": model_cfg.default_guidance,
            "max_pixels": model_cfg.max_pixels,
            "dim_alignment": model_cfg.dim_alignment,
        }

    async def _prepare_request(
        self, body: GenerateRequest, model_name: str
    ) -> _PreparedRequest:
        """Validate and resolve all request parameters shared by generate and create_job."""
        handle = self._get_handle(model_name)
        reference_images = _validate_reference_images(body, model_name)

        if body.reference_images is not None and reference_images is not None:
            backend = await self._get_backend(model_name)
            if backend == "mlx":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "reference_images_unsupported_backend",
                        "message": (
                            "reference_images require backend='diffusers'; "
                            f"current FLUX.2 Klein backend is 'mlx'. "
                            "Set SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_BACKEND=diffusers."
                        ),
                    },
                )

        defaults = self._get_predictor_defaults(model_name)
        steps = body.steps if body.steps is not None else defaults["default_steps"]
        guidance_scale = (
            body.guidance_scale
            if body.guidance_scale is not None
            else defaults["default_guidance"]
        )
        max_pixels = defaults["max_pixels"]
        dim_alignment = defaults["dim_alignment"]

        w_h = body.width * body.height
        if w_h > max_pixels:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": f"width×height ({w_h}) exceeds model max ({max_pixels})",
                },
            )
        if body.width % dim_alignment or body.height % dim_alignment:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": f"{model_name} requires dimensions aligned to {dim_alignment}px",
                },
            )

        check_prompt_safety(body.prompt)

        return _PreparedRequest(
            model_name=model_name,
            handle=handle,
            reference_images=reference_images,
            steps=steps,
            guidance_scale=guidance_scale,
            max_pixels=max_pixels,
            dim_alignment=dim_alignment,
        )

    @images_app.post(
        "/generate",
        response_model=GenerateResponse,
        dependencies=[Depends(require_api_key), Depends(rate_limit)],
    )
    async def generate_image(
        self,
        body: GenerateRequest,
        request: Request,
        api_key: str = Depends(require_api_key),
        idempotency_key: str | None = Depends(idempotent),
    ) -> dict[str, Any]:
        model_name = body.model or cfg.default_image_model

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                return cached

        req = await self._prepare_request(body, model_name)

        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "Received image generation request",
            model=model_name,
            width=body.width,
            height=body.height,
            request_id=request_id,
        )

        start = time.perf_counter()
        image, used_seed = await req.handle.generate.remote(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=body.width,
            height=body.height,
            seed=body.seed,
            reference_images=req.reference_images,
        )
        latency = time.perf_counter() - start

        img_bytes = encode_pil(image, format=body.output_format)
        image_b64 = b64_encode(img_bytes) if body.response_format == "b64_json" else None
        image_url = None

        response = {
            "id": generate_id(),
            "created": int(time.time()),
            "model": model_name,
            "image_b64": image_b64,
            "image_url": image_url,
            "format": body.output_format,
            "width": body.width,
            "height": body.height,
            "seed": used_seed,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "negative_prompt": body.negative_prompt,
            "latency_s": round(latency, 4),
        }

        if idempotency_key and api_key:
            self._idem.put(api_key, idempotency_key, response, _body_hash(body))

        logger.info(
            "image generated",
            id=response["id"],
            model=model_name,
            user=body.user,
            key_prefix=api_key[:8] if api_key else None,
            latency_s=latency,
            reference_images_count=len(body.reference_images) if body.reference_images else 0,
        )

        return response

    @images_app.get(
        "/models",
        response_model=ImageModelsResponse,
    )
    async def images_models(
        self,
        api_key: str = Depends(require_api_key),
    ) -> dict[str, Any]:
        models_info = {}
        for name, handle in self._handles.items():
            info = await handle.info.remote()
            models_info[name] = info
        return {
            "models": models_info,
            "default": cfg.default_image_model,
        }

    @images_app.get(
        "/models/{name}",
        response_model=ImageModelDetailResponse,
    )
    async def images_model_detail(
        self,
        name: str,
        api_key: str = Depends(require_api_key),
    ) -> dict[str, Any]:
        if name not in self._handles:
            avail = list(self._handles.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": (f"Model '{name}' is not enabled. " f"Available: {avail}"),
                },
            )
        info = await self._handles[name].info.remote()
        return {"model": name, "info": info}

    @images_app.post(
        "/jobs",
        status_code=201,
        response_model=JobResponse,
        dependencies=[Depends(require_api_key), Depends(rate_limit)],
    )
    async def create_job(
        self,
        body: GenerateRequest,
        request: Request,
        response: Response,
        api_key: str = Depends(require_api_key),
        idempotency_key: str | None = Depends(idempotent),
    ) -> dict[str, Any]:
        model_name = body.model or cfg.default_image_model

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                prefix = request.scope.get("root_path", "/v1/images")
                response.headers["Location"] = f"{prefix}/jobs/{cached['job_id']}"
                return cached

        req = await self._prepare_request(body, model_name)
        store = self._get_store()

        ref = req.handle.generate.remote(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=body.width,
            height=body.height,
            seed=body.seed,
            reference_images=req.reference_images,
        )

        job_id = await store.submit.remote(
            model=model_name,
            user=body.user,
            api_key=api_key,
        )

        self._refs[job_id] = ref

        self._track_job(job_id, ref, model_name, store)

        job_resp = await store.get.remote(job_id, api_key)

        prefix = request.scope.get("root_path", "/v1/images")
        response.headers["Location"] = f"{prefix}/jobs/{job_id}"

        if idempotency_key and api_key:
            self._idem.put(api_key, idempotency_key, job_resp, _body_hash(body))

        logger.info(
            "job created",
            job_id=job_id,
            model=model_name,
            user=body.user,
            key_prefix=api_key[:8],
            request_id=getattr(request.state, "request_id", "unknown"),
            reference_images_count=len(body.reference_images) if body.reference_images else 0,
        )

        return job_resp

    def _track_job(
        self,
        job_id: str,
        ref: Any,
        model_name: str,
        store: Any,
    ) -> None:
        import ray

        async def _poll() -> None:
            try:
                result = await ref
                await store.set_succeeded.remote(job_id, result)
            except ray.exceptions.TaskCancelledError:
                # Normal path: cancel_job already set status to "canceled" before
                # calling ray.cancel(). For unexpected cancellation (preemption, OOM),
                # mark as failed so the job doesn't stay in "processing" indefinitely.
                status = await store.get_status.remote(job_id)
                if status not in ("canceled", "succeeded", "failed"):
                    await store.set_failed.remote(job_id, "task_cancelled", "task was cancelled")
            except Exception as exc:
                await store.set_failed.remote(job_id, "generation_failed", str(exc))
            finally:
                self._refs.pop(job_id, None)

        task = asyncio.create_task(_poll())
        self._poll_tasks.add(task)
        task.add_done_callback(self._poll_tasks.discard)

    @images_app.get(
        "/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def get_job(
        self,
        job_id: str,
        api_key: str = Depends(require_api_key),
    ) -> dict[str, Any]:
        store = self._get_store()
        result = await store.get.remote(job_id, api_key)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        return result

    @images_app.get(
        "/jobs",
        response_model=JobListResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def list_jobs(
        self,
        api_key: str = Depends(require_api_key),
        page_size: int = Query(20, ge=1, le=100),
        page_token: str | None = Query(None),
        status: str | None = Query(None),
        model: str | None = Query(None),
    ) -> dict[str, Any]:
        store = self._get_store()
        return await store.list_jobs.remote(
            api_key=api_key,
            page_size=page_size,
            page_token=page_token,
            status_filter=status,
            model_filter=model,
        )

    @images_app.delete(
        "/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def cancel_job(
        self,
        job_id: str,
        api_key: str = Depends(require_api_key),
    ) -> dict[str, Any]:
        import ray

        store = self._get_store()
        result = await store.cancel.remote(job_id, api_key)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        ref = self._refs.pop(job_id, None)
        if ref is not None:
            ray.cancel(ref, force=False)
        return result
