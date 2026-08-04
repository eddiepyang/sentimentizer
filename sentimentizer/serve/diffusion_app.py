"""Headless ComfyUI image deployment and HTTP routes."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import orjson
from fastapi import Depends, HTTPException, Query, Request, Response

from sentimentizer import logger
from sentimentizer.diffusion.comfyui import ComfyUICancelled, ComfyUIClient, ComfyUIError
from sentimentizer.diffusion.config import IMAGE_MODEL_CONFIGS, ImageModelConfig
from sentimentizer.diffusion.image_utils import b64_encode, encode_pil, generate_id
from sentimentizer.diffusion.moderation import (
    ImageModerationClient,
    ImageModerationError,
    UnsafeImageError,
)
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


def _model_info(
    name: str,
    model_cfg: ImageModelConfig,
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "error": error,
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": model_cfg.max_pixels,
        "default_steps": model_cfg.default_steps,
        "default_guidance": model_cfg.default_guidance,
        "quantization": model_cfg.quantization,
        "backend": "comfyui",
    }


_BASE_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "VAEDecode",
    "PreviewImage",
}
_MODEL_NODES = {
    "krea_2": {"EmptyLatentImage", "KSampler"},
    "ideogram_4": {
        "EmptyFlux2LatentImage",
        "RandomNoise",
        "KSamplerSelect",
        "Ideogram4Scheduler",
        "CFGOverride",
        "DualModelGuider",
        "SamplerCustomAdvanced",
    },
}


def _choices(node_info: dict[str, Any], node_class: str, input_name: str) -> set[str]:
    try:
        choices = node_info[node_class]["input"]["required"][input_name][0]
    except (KeyError, IndexError, TypeError):
        return set()
    return {str(choice) for choice in choices} if isinstance(choices, list) else set()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    health_check_timeout_s=30,
)
class ComfyUIDeployment:
    """Serializes workflow submissions to a separately managed CUDA process."""

    def __init__(self, model_names: list[str]) -> None:
        self._models = {name: IMAGE_MODEL_CONFIGS[name] for name in model_names}
        self._client = ComfyUIClient(
            cfg.comfyui_base_url,
            timeout_s=cfg.comfyui_timeout_s,
            poll_interval_s=cfg.comfyui_poll_interval_s,
            temp_directory=cfg.comfyui_temp_directory,
        )
        self._generation_lock = asyncio.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._metrics = ServiceMetrics(prefix="comfyui_images")
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        self._client.system_stats()
        for model_name, model_cfg in self._models.items():
            required_nodes = _BASE_NODES | _MODEL_NODES[model_name]
            node_details: dict[str, Any] = {}
            missing_nodes: list[str] = []
            for node_class in sorted(required_nodes):
                info = self._client.node_info(node_class)
                if node_class not in info:
                    missing_nodes.append(node_class)
                else:
                    node_details.update(info)
            if missing_nodes:
                raise RuntimeError(
                    f"ComfyUI is missing nodes required by {model_name}: {missing_nodes}. "
                    "Install a current native ComfyUI release; custom nodes are not required."
                )
            checkpoints = {
                ("UNETLoader", "unet_name"): [
                    model_cfg.transformer,
                    *(
                        [model_cfg.unconditional_transformer]
                        if model_cfg.unconditional_transformer
                        else []
                    ),
                ],
                ("CLIPLoader", "clip_name"): [model_cfg.text_encoder],
                ("VAELoader", "vae_name"): [model_cfg.vae],
            }
            missing_checkpoints: list[str] = []
            for (node_class, input_name), filenames in checkpoints.items():
                available = _choices(node_details, node_class, input_name)
                missing_checkpoints.extend(name for name in filenames if name not in available)
            if missing_checkpoints:
                raise RuntimeError(
                    f"ComfyUI is missing checkpoints required by {model_name}: "
                    f"{missing_checkpoints}"
                )

    async def generate(self, model_name: str, **kwargs: Any) -> dict[str, Any]:
        if model_name not in self._models:
            raise ValueError(f"model is not enabled: {model_name}")
        prompt_id = kwargs.get("prompt_id")
        cancel_event = threading.Event()
        if isinstance(prompt_id, str):
            self._cancel_events[prompt_id] = cancel_event
        kwargs["cancel_event"] = cancel_event
        start = time.perf_counter()
        try:
            async with self._generation_lock:
                result = await asyncio.to_thread(
                    self._client.generate,
                    model_name,
                    self._models[model_name],
                    **kwargs,
                )
        except Exception:
            self._metrics.record_request(time.perf_counter() - start, error=True)
            raise
        finally:
            if isinstance(prompt_id, str):
                self._cancel_events.pop(prompt_id, None)
        latency = time.perf_counter() - start
        self._metrics.record_request(latency)
        return {"image": result.data, "seed": result.seed, "latency_s": latency}

    async def cancel(self, prompt_id: str) -> bool:
        """Cancel a targeted queued or running ComfyUI prompt."""
        cancel_event = self._cancel_events.get(prompt_id)
        if cancel_event is not None:
            cancel_event.set()
        return await asyncio.to_thread(self._client.cancel, prompt_id)

    async def check_health(self) -> None:
        """Fail Ray Serve health checks when the sidecar is unavailable."""
        await asyncio.to_thread(self._client.system_stats)

    async def info(self, model_name: str) -> dict[str, Any]:
        try:
            await self.check_health()
        except Exception as exc:
            logger.warning("comfyui_health_check_failed", error=str(exc))
            return _model_info(
                model_name,
                self._models[model_name],
                status="error",
                error="ComfyUI sidecar unavailable",
            )
        return _model_info(model_name, self._models[model_name], status="loaded")


@dataclass(frozen=True)
class _PreparedRequest:
    model_name: str
    model_config: ImageModelConfig
    steps: int
    guidance_scale: float


images_app = create_fastapi_app(
    title="Sentimentizer Images",
    description="Headless Krea 2 and Ideogram 4 image generation API",
    path_limits={"/": 4 * 1024 * 1024},
)


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=20,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(images_app)
class ImagesDispatcher:
    """Image API front door forwarding work to one headless ComfyUI queue."""

    def __init__(self, comfyui: Any, model_names: list[str]) -> None:
        self._comfyui = comfyui
        self._model_names = set(model_names)
        self._idem = IdempotencyCache(ttl_s=cfg.idempotency_ttl_s)
        self._moderator = (
            ImageModerationClient(
                cfg.image_moderation_url,
                api_key=cfg.image_moderation_api_key,
                timeout_s=cfg.image_moderation_timeout_s,
            )
            if cfg.image_moderation_url
            else None
        )
        self._store: Any = None
        self._refs: dict[str, Any] = {}
        self._poll_tasks: set[asyncio.Task[Any]] = set()

    def _get_store(self) -> Any:
        import ray

        if self._store is None:
            try:
                self._store = ray.get_actor("diffusion_job_store")
            except ValueError:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "job_store_unavailable", "message": "Job store unavailable"},
                ) from None
        return self._store

    def _prepare_request(self, body: GenerateRequest, model_name: str) -> _PreparedRequest:
        if model_name not in self._model_names:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "model_unavailable",
                    "message": (
                        f"Model '{model_name}' is not enabled. "
                        f"Available: {sorted(self._model_names)}"
                    ),
                },
            )
        if body.negative_prompt:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "negative_prompt_unsupported",
                    "message": f"{model_name} uses its native unconditional workflow",
                },
            )
        if body.reference_images is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "reference_images_unsupported",
                    "message": "Krea 2 and Ideogram 4 currently support text-to-image only",
                },
            )
        if body.response_format == "url":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "response_format_unsupported",
                    "message": "url output requires an external storage backend; use b64_json",
                },
            )
        model_config = IMAGE_MODEL_CONFIGS[model_name]
        pixels = body.width * body.height
        if pixels > model_config.max_pixels:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": (
                        f"width×height ({pixels}) exceeds model max ({model_config.max_pixels})"
                    ),
                },
            )
        if body.width % model_config.dim_alignment or body.height % model_config.dim_alignment:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_dimensions",
                    "message": f"{model_name} requires dimensions aligned to 16px",
                },
            )
        check_prompt_safety(body.prompt)
        return _PreparedRequest(
            model_name=model_name,
            model_config=model_config,
            steps=body.steps or model_config.default_steps,
            guidance_scale=(
                body.guidance_scale
                if body.guidance_scale is not None
                else model_config.default_guidance
            ),
        )

    def _submit(
        self,
        body: GenerateRequest,
        prepared: _PreparedRequest,
        *,
        prompt_id: str | None = None,
    ) -> Any:
        return self._comfyui.generate.remote(
            prepared.model_name,
            prompt=body.prompt,
            steps=prepared.steps,
            guidance_scale=prepared.guidance_scale,
            width=body.width,
            height=body.height,
            seed=body.seed,
            prompt_id=prompt_id,
        )

    async def _response(
        self,
        body: GenerateRequest,
        prepared: _PreparedRequest,
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        import PIL.Image

        if self._moderator is not None:
            try:
                await asyncio.to_thread(
                    self._moderator.check,
                    generated["image"],
                    model=prepared.model_name,
                    user=body.user,
                )
            except UnsafeImageError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"code": exc.code, "message": exc.message},
                ) from exc
            except ImageModerationError as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "image_moderation_unavailable",
                        "message": "Generated output could not be safety checked",
                    },
                ) from exc

        with PIL.Image.open(io.BytesIO(generated["image"])) as image:
            image.load()
            image_bytes = encode_pil(image, format=body.output_format)
        return {
            "id": generate_id(),
            "created": int(time.time()),
            "model": prepared.model_name,
            "image_b64": b64_encode(image_bytes),
            "image_url": None,
            "format": body.output_format,
            "width": body.width,
            "height": body.height,
            "seed": generated["seed"],
            "steps": prepared.steps,
            "guidance_scale": prepared.guidance_scale,
            "negative_prompt": None,
            "latency_s": round(float(generated["latency_s"]), 4),
        }

    async def _await_generation(self, ref: Any) -> dict[str, Any]:
        try:
            return await ref
        except Exception as exc:
            raise _backend_http_exception(exc) from exc

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
            body_hash = _body_hash(body)
            self._idem.check_conflict(api_key, idempotency_key, body_hash)
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                return cached
        prepared = self._prepare_request(body, model_name)
        logger.info(
            "received_image_generation_request",
            model=model_name,
            width=body.width,
            height=body.height,
            request_id=getattr(request.state, "request_id", "unknown"),
        )
        generated = await self._await_generation(self._submit(body, prepared))
        result = await self._response(body, prepared, generated)
        if idempotency_key and api_key:
            self._idem.put(api_key, idempotency_key, result, _body_hash(body))
        return result

    @images_app.get("/models", response_model=ImageModelsResponse)
    async def images_models(self, api_key: str = Depends(require_api_key)) -> dict[str, Any]:
        models = {name: await self._comfyui.info.remote(name) for name in sorted(self._model_names)}
        return {"models": models, "default": cfg.default_image_model}

    @images_app.get("/models/{name}", response_model=ImageModelDetailResponse)
    async def images_model_detail(
        self, name: str, api_key: str = Depends(require_api_key)
    ) -> dict[str, Any]:
        if name not in self._model_names:
            raise HTTPException(
                status_code=400,
                detail={"code": "model_unavailable", "message": f"Model '{name}' is not enabled"},
            )
        return {"model": name, "info": await self._comfyui.info.remote(name)}

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
            body_hash = _body_hash(body)
            self._idem.check_conflict(api_key, idempotency_key, body_hash)
            cached = self._idem.get(api_key, idempotency_key)
            if cached is not None:
                prefix = request.scope.get("root_path", "/v1/images")
                response.headers["Location"] = f"{prefix}/jobs/{cached['job_id']}"
                return cached
        prepared = self._prepare_request(body, model_name)
        store = self._get_store()
        prompt_id = str(uuid.uuid4())
        job_id = await store.submit.remote(
            model=model_name,
            user=body.user,
            api_key=api_key,
            backend_id=prompt_id,
        )
        ref = self._submit(body, prepared, prompt_id=prompt_id)
        self._refs[job_id] = ref
        self._track_job(job_id, ref, body, prepared, store)
        job_response = await store.get.remote(job_id, api_key)
        prefix = request.scope.get("root_path", "/v1/images")
        response.headers["Location"] = f"{prefix}/jobs/{job_id}"
        if idempotency_key and api_key:
            self._idem.put(api_key, idempotency_key, job_response, _body_hash(body))
        return job_response

    def _track_job(
        self,
        job_id: str,
        ref: Any,
        body: GenerateRequest,
        prepared: _PreparedRequest,
        store: Any,
    ) -> None:
        import ray

        async def _poll() -> None:
            try:
                generated = await ref
                result = await self._response(body, prepared, generated)
                await store.set_succeeded.remote(job_id, result)
            except (ray.exceptions.TaskCancelledError, ComfyUICancelled):
                status = await store.get_status.remote(job_id)
                if status not in ("canceled", "succeeded", "failed"):
                    await store.set_failed.remote(job_id, "task_cancelled", "task was cancelled")
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                await store.set_failed.remote(
                    job_id,
                    str(detail.get("code") or "generation_failed"),
                    str(detail.get("message") or "Image generation failed"),
                )
            except Exception as exc:
                backend_error = _backend_http_exception(exc)
                detail = backend_error.detail
                await store.set_failed.remote(
                    job_id,
                    str(detail["code"]),
                    str(detail["message"]),
                )
            finally:
                self._refs.pop(job_id, None)

        task = asyncio.create_task(_poll())
        self._poll_tasks.add(task)
        task.add_done_callback(self._poll_tasks.discard)

    @images_app.get(
        "/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)]
    )
    async def get_job(self, job_id: str, api_key: str = Depends(require_api_key)) -> dict[str, Any]:
        result = await self._get_store().get.remote(job_id, api_key)
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
        return await self._get_store().list_jobs.remote(
            api_key=api_key,
            page_size=page_size,
            page_token=page_token,
            status_filter=status,
            model_filter=model,
        )

    @images_app.delete(
        "/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)]
    )
    async def cancel_job(
        self, job_id: str, api_key: str = Depends(require_api_key)
    ) -> dict[str, Any]:
        store = self._get_store()
        current = await store.get.remote(job_id, api_key)
        if current is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "job_not_found", "message": "Job not found"},
            )
        if current["status"] != "processing":
            return current

        prompt_id = await store.get_backend_id.remote(job_id, api_key)
        if prompt_id:
            try:
                await self._comfyui.cancel.remote(prompt_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "backend_cancel_failed",
                        "message": "Could not cancel the ComfyUI prompt; retry the request",
                    },
                ) from exc

        ref = self._refs.get(job_id)
        if ref is not None and not prompt_id:
            self._refs.pop(job_id, None)
            ref.cancel()
        result = await store.cancel.remote(job_id, api_key)
        return result


def _backend_http_exception(exc: Exception) -> HTTPException:
    """Translate ComfyUI and Ray transport failures into stable API errors."""
    cause: BaseException = exc
    try:
        import ray

        if isinstance(exc, ray.exceptions.RayTaskError):
            cause = exc.as_instanceof_cause()
        if isinstance(exc, ray.exceptions.RayActorError):
            return HTTPException(
                status_code=503,
                detail={
                    "code": "image_backend_unavailable",
                    "message": "Image backend unavailable",
                },
            )
    except (ImportError, AttributeError):
        pass

    message = str(cause)
    if isinstance(cause, ComfyUICancelled):
        return HTTPException(
            status_code=409,
            detail={"code": "generation_cancelled", "message": "Image generation was cancelled"},
        )
    if isinstance(cause, ComfyUIError) or "ComfyUI" in message:
        if "timed out" in message.lower():
            return HTTPException(
                status_code=504,
                detail={"code": "image_backend_timeout", "message": "Image backend timed out"},
            )
        return HTTPException(
            status_code=502,
            detail={
                "code": "image_backend_error",
                "message": "Image backend rejected or failed the workflow",
            },
        )
    return HTTPException(
        status_code=503,
        detail={"code": "image_backend_unavailable", "message": "Image backend unavailable"},
    )
