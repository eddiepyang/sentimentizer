"""Headless ComfyUI client and native Krea 2 / Ideogram 4 workflows."""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentimentizer.diffusion.config import ImageModelConfig


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI rejects or fails an image workflow."""


class ComfyUICancelled(ComfyUIError):
    """Raised when a ComfyUI prompt is cancelled while executing."""


@dataclass(frozen=True)
class GeneratedImage:
    """Image bytes returned by a completed ComfyUI prompt."""

    data: bytes
    seed: int


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_krea_2_workflow(
    config: ImageModelConfig,
    *,
    prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    """Build the official minimal Krea 2 Turbo INT8 ConvRot workflow."""
    return {
        "1": _node(
            "UNETLoader",
            unet_name=config.transformer,
            weight_dtype="default",
        ),
        "2": _node(
            "CLIPLoader",
            clip_name=config.text_encoder,
            type="krea2",
            device="default",
        ),
        "3": _node("VAELoader", vae_name=config.vae),
        "4": _node("CLIPTextEncode", text=prompt, clip=["2", 0]),
        "5": _node("ConditioningZeroOut", conditioning=["4", 0]),
        "6": _node("EmptyLatentImage", width=width, height=height, batch_size=1),
        "7": _node(
            "KSampler",
            model=["1", 0],
            positive=["4", 0],
            negative=["5", 0],
            latent_image=["6", 0],
            seed=seed,
            steps=steps,
            cfg=guidance_scale,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        ),
        "8": _node("VAEDecode", samples=["7", 0], vae=["3", 0]),
        "preview_image": _node("PreviewImage", images=["8", 0]),
    }


def build_ideogram_4_workflow(
    config: ImageModelConfig,
    *,
    prompt: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    """Build the official minimal Ideogram 4 INT8 ConvRot workflow."""
    mu, std = _ideogram_schedule(steps)
    return {
        "1": _node("VAELoader", vae_name=config.vae),
        "2": _node(
            "UNETLoader",
            unet_name=config.transformer,
            weight_dtype="default",
        ),
        "3": _node(
            "UNETLoader",
            unet_name=config.unconditional_transformer,
            weight_dtype="default",
        ),
        "4": _node(
            "CLIPLoader",
            clip_name=config.text_encoder,
            type="ideogram4",
            device="default",
        ),
        "5": _node("CLIPTextEncode", text=prompt, clip=["4", 0]),
        "6": _node("ConditioningZeroOut", conditioning=["5", 0]),
        "7": _node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1),
        "8": _node("RandomNoise", noise_seed=seed),
        "9": _node("KSamplerSelect", sampler_name="euler"),
        "10": _node(
            "Ideogram4Scheduler",
            steps=steps,
            width=width,
            height=height,
            mu=mu,
            std=std,
        ),
        "11": _node(
            "CFGOverride",
            model=["2", 0],
            cfg=3.0,
            start_percent=0.7,
            end_percent=1.0,
        ),
        "12": _node(
            "DualModelGuider",
            model=["11", 0],
            model_negative=["3", 0],
            positive=["5", 0],
            cfg=guidance_scale,
            negative=["6", 0],
        ),
        "13": _node(
            "SamplerCustomAdvanced",
            noise=["8", 0],
            guider=["12", 0],
            sampler=["9", 0],
            sigmas=["10", 0],
            latent_image=["7", 0],
        ),
        "14": _node("VAEDecode", samples=["13", 0], vae=["1", 0]),
        "preview_image": _node("PreviewImage", images=["14", 0]),
    }


def _ideogram_schedule(steps: int) -> tuple[float, float]:
    if steps <= 12:
        return 0.5, 1.75
    if steps >= 40:
        return 0.0, 1.5
    return 0.0, 1.75


class ComfyUIClient:
    """Small synchronous client for ComfyUI's queue/history HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 600.0,
        poll_interval_s: float = 0.25,
        temp_directory: str | Path | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("comfyui_base_url must be an http(s) URL")
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("ComfyUI timeout and poll interval must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.temp_directory = Path(temp_directory).resolve() if temp_directory else None
        self.client_id = str(uuid.uuid4())

    def system_stats(self) -> dict[str, Any]:
        """Return ComfyUI system/device details and verify reachability."""
        return self._request_json("GET", "/system_stats")

    def node_info(self, node_class: str) -> dict[str, Any]:
        """Return metadata for one registered ComfyUI node class."""
        quoted = urllib.parse.quote(node_class, safe="")
        return self._request_json("GET", f"/object_info/{quoted}")

    def generate(
        self,
        model_name: str,
        config: ImageModelConfig,
        *,
        prompt: str,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: int | None,
        prompt_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> GeneratedImage:
        """Queue a workflow, wait for completion, then fetch its first image."""
        used_seed = seed if seed is not None else secrets.randbelow(2**63)
        used_prompt_id = prompt_id or str(uuid.uuid4())
        builders = {
            "krea_2": build_krea_2_workflow,
            "ideogram_4": build_ideogram_4_workflow,
        }
        try:
            builder = builders[model_name]
        except KeyError as exc:
            raise ValueError(f"unsupported image model: {model_name}") from exc
        workflow = builder(
            config,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=used_seed,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise ComfyUICancelled(f"ComfyUI prompt {used_prompt_id} was cancelled")
        queued = self._request_json(
            "POST",
            "/prompt",
            {
                "prompt": workflow,
                "client_id": self.client_id,
                "prompt_id": used_prompt_id,
            },
        )
        queued_prompt_id = queued.get("prompt_id")
        if not isinstance(queued_prompt_id, str) or not queued_prompt_id:
            raise ComfyUIError(f"ComfyUI returned no prompt_id: {queued!r}")
        if queued_prompt_id != used_prompt_id:
            raise ComfyUIError(
                f"ComfyUI changed prompt_id from {used_prompt_id!r} to {queued_prompt_id!r}"
            )
        try:
            if cancel_event is not None and cancel_event.is_set():
                self.cancel(queued_prompt_id)
                raise ComfyUICancelled(f"ComfyUI prompt {queued_prompt_id} was cancelled")
            descriptor = self._wait_for_image(queued_prompt_id, cancel_event=cancel_event)
            query = urllib.parse.urlencode(descriptor)
            image = self._request_bytes("GET", f"/view?{query}")
            self._remove_temp_image(descriptor)
            return GeneratedImage(image, used_seed)
        finally:
            self._delete_history(queued_prompt_id)

    def cancel(self, prompt_id: str) -> bool:
        """Cancel one queued or running prompt without affecting other work."""
        quoted = urllib.parse.quote(prompt_id, safe="")
        result = self._request_json("POST", f"/api/jobs/{quoted}/cancel", {})
        return result.get("cancelled") is True

    def _delete_history(self, prompt_id: str) -> None:
        """Best-effort cleanup of completed workflow metadata."""
        with suppress(ComfyUIError):
            self._request(
                "POST",
                "/history",
                json.dumps({"delete": [prompt_id]}).encode(),
                "application/json",
            )

    def _remove_temp_image(self, descriptor: dict[str, str]) -> None:
        """Delete one fetched PreviewImage artifact from the shared temp path."""
        if self.temp_directory is None:
            return
        if descriptor.get("type") != "temp":
            raise ComfyUIError("ComfyUI returned a non-temporary image artifact")
        candidate = (
            self.temp_directory / descriptor.get("subfolder", "") / descriptor["filename"]
        ).resolve()
        if not candidate.is_relative_to(self.temp_directory):
            raise ComfyUIError("ComfyUI returned an unsafe temporary image path")
        try:
            candidate.unlink()
        except OSError as exc:
            raise ComfyUIError(f"Could not remove ComfyUI temporary image {candidate}") from exc

    def _wait_for_image(
        self,
        prompt_id: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, str]:
        deadline = time.monotonic() + self.timeout_s
        path = f"/history/{urllib.parse.quote(prompt_id, safe='')}"
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self.cancel(prompt_id)
                raise ComfyUICancelled(f"ComfyUI prompt {prompt_id} was cancelled")
            history = self._request_json("GET", path)
            entry = history.get(prompt_id)
            if isinstance(entry, dict):
                outputs = entry.get("outputs", {})
                preview_output = (
                    outputs.get("preview_image", {}) if isinstance(outputs, dict) else {}
                )
                images = (
                    preview_output.get("images", []) if isinstance(preview_output, dict) else []
                )
                if images:
                    image = images[0]
                    if isinstance(image, dict):
                        return {
                            "filename": str(image["filename"]),
                            "subfolder": str(image.get("subfolder", "")),
                            "type": str(image.get("type", "temp")),
                        }
                status = entry.get("status", {})
                if isinstance(status, dict) and status.get("status_str") == "error":
                    if _was_cancelled(entry):
                        raise ComfyUICancelled(f"ComfyUI prompt {prompt_id} was cancelled")
                    raise ComfyUIError(_execution_error(entry))
            time.sleep(self.poll_interval_s)
        self.cancel(prompt_id)
        raise ComfyUIError(f"ComfyUI prompt {prompt_id} timed out after {self.timeout_s:g}s")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        raw = self._request(method, path, data, "application/json")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComfyUIError(f"ComfyUI returned invalid JSON for {path}") from exc
        if not isinstance(value, dict):
            raise ComfyUIError(f"ComfyUI returned non-object JSON for {path}")
        return value

    def _request_bytes(self, method: str, path: str) -> bytes:
        return self._request(method, path, None, None)

    def _request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        content_type: str | None,
    ) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ComfyUIError(f"ComfyUI HTTP {exc.code} for {path}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ComfyUIError(f"Cannot reach ComfyUI at {self.base_url}: {reason}") from exc


def _execution_error(entry: dict[str, Any]) -> str:
    status = entry.get("status", {})
    messages = status.get("messages", []) if isinstance(status, dict) else []
    for message in reversed(messages):
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error":
            detail = message[1]
            if isinstance(detail, dict):
                return str(
                    detail.get("exception_message") or detail.get("exception_type") or detail
                )
    return "ComfyUI workflow execution failed"


def _was_cancelled(entry: dict[str, Any]) -> bool:
    status = entry.get("status", {})
    messages = status.get("messages", []) if isinstance(status, dict) else []
    return any(
        isinstance(message, list | tuple)
        and len(message) == 2
        and message[0] == "execution_interrupted"
        for message in messages
    )
