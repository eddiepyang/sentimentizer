"""Diffusion serving: dispatcher, SD/FLUX deployments, and image generation routes.

Uses the Ray Serve composition pattern: a lightweight CPU dispatcher
holds DeploymentHandles to GPU-backed model deployments. Routes are
attached to the dispatcher via ``@app.post`` / ``@app.get`` on the same
FastAPI app from ``sentimentizer/serve/app.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch
from fastapi import Depends, HTTPException, Query, Request, Response

from sentimentizer import logger
from sentimentizer.diffusion.predictor import (
    FluxPredictor,
    SD35Predictor,
    SDPredictor,
    _b64,
    _encode_pil,
    _generate_id,
)
from sentimentizer.serve.app import (
    cfg,  # noqa: E402
    create_fastapi_app,
)
from sentimentizer.serve.base import ServiceMetrics, serve
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


def _body_hash(body: Any) -> str:
    import hashlib

    import orjson

    if hasattr(body, "model_dump"):
        data = body.model_dump(mode="python")
    elif hasattr(body, "dict"):
        data = body.dict()
    else:
        data = body
    raw = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:32]


_num_gpus = 1 if torch.cuda.is_available() else 0

images_app = create_fastapi_app(
    title="Sentimentizer Images",
    description="Image generation API (Stable Diffusion / FLUX)",
)


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 2, "num_gpus": _num_gpus},
)
class SDDeployment:
    def __init__(self) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import SD_DEFAULT_CONFIG

        model_cfg = SD_DEFAULT_CONFIG
        if cfg.sd_model_id:
            model_cfg = replace(SD_DEFAULT_CONFIG, model_id=cfg.sd_model_id)
        self.predictor = SDPredictor(model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="sd")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        return await asyncio.to_thread(self.predictor.generate, **kwargs)

    def info(self) -> dict[str, Any]:
        return self.predictor.model_info()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={"num_cpus": 4, "num_gpus": _num_gpus},
)
class FluxDeployment:
    def __init__(self) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import FLUX_DEFAULT_CONFIG

        model_cfg = FLUX_DEFAULT_CONFIG
        if cfg.flux_model_path:
            model_cfg = replace(FLUX_DEFAULT_CONFIG, model_path=cfg.flux_model_path)
        self.predictor = FluxPredictor(model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="flux")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        return await asyncio.to_thread(self.predictor.generate, **kwargs)

    def info(self) -> dict[str, Any]:
        return self.predictor.model_info()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_cpus": 2, "num_gpus": _num_gpus},
)
class SD35Deployment:
    def __init__(self) -> None:
        from dataclasses import replace

        from sentimentizer.diffusion.config import SD35_DEFAULT_CONFIG

        model_cfg = SD35_DEFAULT_CONFIG
        if cfg.sd35_model_id:
            model_cfg = replace(SD35_DEFAULT_CONFIG, model_id=cfg.sd35_model_id)
        self.predictor = SD35Predictor(model_cfg)
        self.predictor.warmup()
        self._metrics = ServiceMetrics(prefix="sd35")

    async def generate(self, **kwargs: Any) -> tuple[Any, int]:
        return await asyncio.to_thread(self.predictor.generate, **kwargs)

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
        sd: Any = None,
        flux: Any = None,
        sd35: Any = None,
    ) -> None:
        self._handles: dict[str, Any] = {}
        if sd is not None:
            self._handles["sd"] = sd
        if flux is not None:
            self._handles["flux"] = flux
        if sd35 is not None:
            self._handles["sd35"] = sd35
        self._idem = IdempotencyCache(ttl_s=cfg.idempotency_ttl_s)
        self._store: Any = None
        self._refs: dict[str, Any] = {}

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
        from sentimentizer.diffusion.config import (
            FLUX_DEFAULT_CONFIG,
            SD35_DEFAULT_CONFIG,
            SD_DEFAULT_CONFIG,
        )

        defaults_map = {
            "sd": SD_DEFAULT_CONFIG,
            "flux": FLUX_DEFAULT_CONFIG,
            "sd35": SD35_DEFAULT_CONFIG,
        }
        model_cfg = defaults_map.get(model)
        if model_cfg is None:
            return {}
        return {
            "default_steps": model_cfg.default_steps,
            "default_guidance": model_cfg.default_guidance,
            "max_pixels": model_cfg.max_pixels,
            "dim_alignment": model_cfg.dim_alignment,
        }

    @images_app.post(
        "/",
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
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "Received image generation request",
            model=model_name,
            width=body.width,
            height=body.height,
            request_id=request_id,
        )
        handle = self._get_handle(model_name)

        defaults = self._get_predictor_defaults(model_name)
        steps = body.steps if body.steps is not None else defaults.get("default_steps", 30)
        guidance_scale = (
            body.guidance_scale
            if body.guidance_scale is not None
            else defaults.get("default_guidance", 7.5)
        )
        max_pixels = defaults.get("max_pixels", 1048576)
        dim_alignment = defaults.get("dim_alignment", 8)

        w_h = body.width * body.height
        if w_h > max_pixels:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": (f"width×height ({w_h}) exceeds " f"model max ({max_pixels})"),
                },
            )
        if body.width % dim_alignment or body.height % dim_alignment:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": (
                        f"{model_name} requires dimensions " f"aligned to {dim_alignment}px"
                    ),
                },
            )

        check_prompt_safety(body.prompt)

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        image, used_seed = await handle.generate.remote(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=body.width,
            height=body.height,
            seed=body.seed,
        )
        latency = time.perf_counter() - start

        img_bytes = _encode_pil(image, format=body.output_format)
        image_b64 = _b64(img_bytes) if body.response_format == "b64_json" else None
        image_url = None

        response = {
            "id": _generate_id(),
            "created": int(time.time()),
            "model": model_name,
            "image_b64": image_b64,
            "image_url": image_url,
            "format": body.output_format,
            "width": body.width,
            "height": body.height,
            "seed": used_seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
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
        handle = self._get_handle(model_name)

        defaults = self._get_predictor_defaults(model_name)
        steps = body.steps if body.steps is not None else defaults.get("default_steps", 30)
        guidance_scale = (
            body.guidance_scale
            if body.guidance_scale is not None
            else defaults.get("default_guidance", 7.5)
        )
        max_pixels = defaults.get("max_pixels", 1048576)
        dim_alignment = defaults.get("dim_alignment", 8)

        w_h = body.width * body.height
        if w_h > max_pixels:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": (f"width\u00d7height ({w_h}) exceeds " f"model max ({max_pixels})"),
                },
            )
        if body.width % dim_alignment or body.height % dim_alignment:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": (
                        f"{model_name} requires dimensions " f"aligned to {dim_alignment}px"
                    ),
                },
            )

        check_prompt_safety(body.prompt)

        store = self._get_store()

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                response.headers["Location"] = f"/v1/images/jobs/{cached['job_id']}"
                return cached

        ref = handle.generate.remote(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=body.width,
            height=body.height,
            seed=body.seed,
        )

        job_id = await store.submit.remote(
            model=model_name,
            user=body.user,
            api_key=api_key,
        )

        self._refs[job_id] = ref

        self._track_job(job_id, ref, model_name, store)

        job_resp = await store.get.remote(job_id, api_key)

        response.headers["Location"] = f"/v1/images/jobs/{job_id}"

        if idempotency_key and api_key:
            self._idem.put(api_key, idempotency_key, job_resp, _body_hash(body))

        logger.info(
            "job created",
            job_id=job_id,
            model=model_name,
            user=body.user,
            key_prefix=api_key[:8],
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
                pass
            except Exception as exc:
                await store.set_failed.remote(job_id, "generation_failed", str(exc))
            finally:
                self._refs.pop(job_id, None)

        import asyncio

        asyncio.create_task(_poll())

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
