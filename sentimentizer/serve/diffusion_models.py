"""Pydantic request/response models for the image generation API.

Follows the same conventions as sentimentizer/serve/models.py:
  - ``model_config = {"json_schema_extra": {"examples": [...]}}`` for Swagger docs
  - ``Annotated[str, Field(min_length=..., max_length=...)]`` for validation
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model: str | None = Field(
        None,
        description="flux2_klein, sd35, or sdxl_<slot>; default: cfg.default_image_model",
    )
    negative_prompt: str | None = Field(None, max_length=2000)
    steps: int | None = Field(None, ge=1, le=100, description="default depends on model")
    guidance_scale: float | None = Field(None, ge=0.0, le=20.0)
    width: int = Field(
        1024,
        ge=256,
        le=2048,
        multiple_of=8,
    )
    height: int = Field(
        1024,
        ge=256,
        le=2048,
        multiple_of=8,
    )
    seed: int | None = None
    response_format: Literal["b64_json", "url"] = "b64_json"
    output_format: Literal["png", "webp", "jpeg"] = "png"
    user: str | None = Field(
        None,
        max_length=128,
        description="opaque end-user id for abuse tracking",
    )
    reference_images: list[str] | None = Field(
        None,
        description=(
            "Base64-encoded reference images (raw base64 or data:image/<fmt>;base64,…). "
            "FLUX.2 Klein only. Up to 2 images, each ≤ 512×512 after decoding (262,144 pixels)."
        ),
    )

    @field_validator("reference_images", mode="before")
    @classmethod
    def validate_reference_images(cls, v: Any) -> Any:
        if v == []:
            return None
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                raise ValueError("reference_images must be a list of non-empty strings")
            if len(v) > 2:
                raise ValueError("at most 2 reference images allowed")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "a red apple on a wooden table",
                    "model": "sd35",
                    "width": 1024,
                    "height": 1024,
                    "output_format": "png",
                },
                {
                    "prompt": "a cinematic portrait of an astronaut",
                    "negative_prompt": "blurry, low quality",
                    "model": "flux2_klein",
                    "steps": 28,
                    "guidance_scale": 3.5,
                    "width": 1024,
                    "height": 1024,
                    "seed": 42,
                    "response_format": "b64_json",
                    "output_format": "webp",
                },
            ]
        }
    }


class GenerateResponse(BaseModel):
    id: str
    created: int
    model: str
    image_b64: str | None = None
    image_url: str | None = None
    format: str
    width: int
    height: int
    seed: int
    steps: int
    guidance_scale: float
    negative_prompt: str | None = None
    latency_s: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "img_ABCDEFGHIJKL",
                    "created": 1700000000,
                    "model": "sd35",
                    "image_b64": "...",
                    "image_url": None,
                    "format": "png",
                    "width": 1024,
                    "height": 1024,
                    "seed": 42,
                    "steps": 30,
                    "guidance_scale": 7.5,
                    "negative_prompt": None,
                    "latency_s": 2.5,
                }
            ]
        }
    }


class ImageModelInfo(BaseModel):
    name: str
    status: Literal["loaded", "not_loaded", "error"]
    error: str | None = None
    max_width: int
    max_height: int
    max_pixels: int
    default_steps: int
    default_guidance: float
    quantization: str | None = None
    backend: str = "diffusers"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "flux2_klein",
                    "status": "loaded",
                    "max_width": 1024,
                    "max_height": 1024,
                    "max_pixels": 1048576,
                    "default_steps": 4,
                    "default_guidance": 0.0,
                    "quantization": None,
                    "backend": "diffusers",
                }
            ]
        }
    }


class ImageModelsResponse(BaseModel):
    models: dict[str, ImageModelInfo]
    default: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "models": {
                        "flux2_klein": {
                            "name": "flux2_klein",
                            "status": "loaded",
                            "max_width": 1024,
                            "max_height": 1024,
                            "max_pixels": 1048576,
                            "default_steps": 4,
                            "default_guidance": 0.0,
                            "backend": "diffusers",
                        }
                    },
                    "default": "flux2_klein",
                }
            ]
        }
    }


class ImageModelDetailResponse(BaseModel):
    model: str
    info: ImageModelInfo

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "flux2_klein",
                    "info": {
                        "name": "flux2_klein",
                        "status": "loaded",
                        "max_width": 1024,
                        "max_height": 1024,
                        "max_pixels": 1048576,
                        "default_steps": 4,
                        "default_guidance": 0.0,
                        "backend": "diffusers",
                    },
                }
            ]
        }
    }


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "succeeded", "failed", "canceled"]
    created: int
    updated: int
    model: str
    user: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job_ABCDEFGHIJKLMNOP",
                    "status": "queued",
                    "created": 1700000000,
                    "updated": 1700000000,
                    "model": "flux2_klein",
                    "user": None,
                }
            ]
        }
    }


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    next_page_token: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "jobs": [
                        {
                            "job_id": "job_ABCDEFGHIJKLMNOP",
                            "status": "processing",
                            "created": 1700000000,
                            "updated": 1700000000,
                            "model": "flux2_klein",
                        }
                    ],
                    "next_page_token": None,
                }
            ]
        }
    }
