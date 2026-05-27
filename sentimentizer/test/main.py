"""Pure-FastAPI image generation API — comparison sketch.

Mirrors the HTTP surface of ``sentimentizer/serve/diffusion_app.py`` without
Ray Serve. Demonstrates the idiomatic current-FastAPI shape: ``lifespan`` for
long-lived state, module-level routes that delegate to a singleton service
held on ``app.state``, and pure-Python dispatch via ``asyncio.to_thread``.

Only SD 3.5 is wired up at startup. Requests for ``model=sd`` or
``model=flux`` will return 400 from ``_get_handle``.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Annotated, Any

import orjson
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response

from sentimentizer import logger
from sentimentizer.diffusion.config import (
    FLUX_DEFAULT_CONFIG,
    SD35_DEFAULT_CONFIG,
    SD_DEFAULT_CONFIG,
)
from sentimentizer.diffusion.job_store import JobStoreLogic
from sentimentizer.diffusion.predictor import (
    SD35Predictor,
    _b64,
    _encode_pil,
    _generate_id,
)
from sentimentizer.serve.config import load_serve_config
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

cfg = load_serve_config()


def _body_hash(body: Any) -> str:
    if hasattr(body, "model_dump"):
        data = body.model_dump(mode="python")
    elif hasattr(body, "dict"):
        data = body.dict()
    else:
        data = body
    raw = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:32]


class ImagesDispatcher:
    """Stateful service that owns predictor handles, idempotency cache,
    job store, and in-flight generation tasks. No FastAPI coupling."""

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
        self._store = JobStoreLogic(ttl_s=cfg.job_ttl_s)
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def _get_handle(self, model: str) -> Any:
        if model not in self._handles:
            avail = list(self._handles.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": f"Model '{model}' is not enabled. Available: {avail}",
                },
            )
        return self._handles[model]

    def _get_predictor_defaults(self, model: str) -> dict[str, Any]:
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

    def _validate_dimensions(self, body: GenerateRequest, model_name: str) -> tuple[int, float]:
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
        return steps, guidance_scale

    async def generate_image(
        self,
        body: GenerateRequest,
        request: Request,
        api_key: str,
        idempotency_key: str | None,
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
        steps, guidance_scale = self._validate_dimensions(body, model_name)

        check_prompt_safety(body.prompt)

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        image, used_seed = await asyncio.to_thread(
            handle.generate,
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

        response = {
            "id": _generate_id(),
            "created": int(time.time()),
            "model": model_name,
            "image_b64": image_b64,
            "image_url": None,
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

    async def images_models(self) -> dict[str, Any]:
        models_info: dict[str, Any] = {}
        for name, handle in self._handles.items():
            models_info[name] = await asyncio.to_thread(handle.model_info)
        return {"models": models_info, "default": cfg.default_image_model}

    async def images_model_detail(self, name: str) -> dict[str, Any]:
        if name not in self._handles:
            avail = list(self._handles.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": f"Model '{name}' is not enabled. Available: {avail}",
                },
            )
        info = await asyncio.to_thread(self._handles[name].model_info)
        return {"model": name, "info": info}

    async def create_job(
        self,
        body: GenerateRequest,
        response: Response,
        api_key: str,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        model_name = body.model or cfg.default_image_model
        handle = self._get_handle(model_name)
        steps, guidance_scale = self._validate_dimensions(body, model_name)

        check_prompt_safety(body.prompt)

        if idempotency_key and api_key:
            self._idem.check_conflict(api_key, idempotency_key, _body_hash(body))
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                response.headers["Location"] = f"/v1/images/jobs/{cached['job_id']}"
                return cached

        task = asyncio.create_task(
            asyncio.to_thread(
                handle.generate,
                prompt=body.prompt,
                negative_prompt=body.negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                width=body.width,
                height=body.height,
                seed=body.seed,
            )
        )

        job_id = self._store.submit(model=model_name, user=body.user, api_key=api_key)
        self._tasks[job_id] = task
        self._track_job(job_id, task)

        job_resp = self._store.get(job_id, api_key)
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

    def _track_job(self, job_id: str, task: asyncio.Task[Any]) -> None:
        async def _poll() -> None:
            try:
                result = await task
                self._store.set_succeeded(job_id, result)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._store.set_failed(job_id, "generation_failed", str(exc))
            finally:
                self._tasks.pop(job_id, None)

        asyncio.create_task(_poll())

    async def get_job(self, job_id: str, api_key: str) -> dict[str, Any]:
        result = self._store.get(job_id, api_key)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        return result

    async def list_jobs(
        self,
        api_key: str,
        page_size: int,
        page_token: str | None,
        status_filter: str | None,
        model_filter: str | None,
    ) -> dict[str, Any]:
        return self._store.list_jobs(
            api_key=api_key,
            page_size=page_size,
            page_token=page_token,
            status_filter=status_filter,
            model_filter=model_filter,
        )

    async def cancel_job(self, job_id: str, api_key: str) -> dict[str, Any]:
        result = self._store.cancel(job_id, api_key)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        task = self._tasks.pop(job_id, None)
        if task is not None:
            task.cancel()
        return result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    overrides: dict[str, Any] = {}
    if cfg.sd35_model_id:
        overrides["model_id"] = cfg.sd35_model_id
    if cfg.sd35_cpu_offload:
        overrides["cpu_offload"] = cfg.sd35_cpu_offload
    model_cfg = replace(SD35_DEFAULT_CONFIG, **overrides) if overrides else SD35_DEFAULT_CONFIG
    sd35 = SD35Predictor(model_cfg)
    sd35.warmup()
    app.state.dispatcher = ImagesDispatcher(sd35=sd35)
    logger.info("dispatcher ready", models=list(app.state.dispatcher._handles.keys()))
    try:
        yield
    finally:
        for task in list(app.state.dispatcher._tasks.values()):
            task.cancel()


def get_dispatcher(request: Request) -> ImagesDispatcher:
    return request.app.state.dispatcher


ApiKey = Annotated[str, Depends(require_api_key)]
IdemKey = Annotated[str | None, Depends(idempotent)]
Dispatcher = Annotated[ImagesDispatcher, Depends(get_dispatcher)]

router = APIRouter(prefix="/v1/images")


@router.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(rate_limit)],
)
async def generate(
    body: GenerateRequest,
    request: Request,
    api_key: ApiKey,
    idempotency_key: IdemKey,
    d: Dispatcher,
) -> dict[str, Any]:
    return await d.generate_image(body, request, api_key, idempotency_key)


@router.get(
    "/models",
    response_model=ImageModelsResponse,
    dependencies=[Depends(require_api_key)],
)
async def list_models(d: Dispatcher) -> dict[str, Any]:
    return await d.images_models()


@router.get(
    "/models/{name}",
    response_model=ImageModelDetailResponse,
    dependencies=[Depends(require_api_key)],
)
async def model_detail(name: str, d: Dispatcher) -> dict[str, Any]:
    return await d.images_model_detail(name)


@router.post(
    "/jobs",
    status_code=201,
    response_model=JobResponse,
    dependencies=[Depends(rate_limit)],
)
async def create_job(
    body: GenerateRequest,
    response: Response,
    api_key: ApiKey,
    idempotency_key: IdemKey,
    d: Dispatcher,
) -> dict[str, Any]:
    return await d.create_job(body, response, api_key, idempotency_key)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    api_key: ApiKey,
    d: Dispatcher,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    page_token: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    return await d.list_jobs(api_key, page_size, page_token, status, model)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, api_key: ApiKey, d: Dispatcher) -> dict[str, Any]:
    return await d.get_job(job_id, api_key)


@router.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str, api_key: ApiKey, d: Dispatcher) -> dict[str, Any]:
    return await d.cancel_job(job_id, api_key)


app = FastAPI(lifespan=lifespan, title="Sentimentizer Images (pure FastAPI)")
app.include_router(router)
