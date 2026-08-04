"""Tests for native headless ComfyUI workflows."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from sentimentizer.diffusion.comfyui import (
    ComfyUICancelled,
    ComfyUIClient,
    ComfyUIError,
    _ideogram_schedule,
    build_ideogram_4_workflow,
    build_krea_2_workflow,
)
from sentimentizer.diffusion.config import IDEOGRAM_4_CONFIG, KREA_2_CONFIG
from sentimentizer.diffusion.moderation import (
    ImageModerationClient,
    ImageModerationError,
    UnsafeImageError,
)
from sentimentizer.serve.config import ServeConfig
from sentimentizer.serve.diffusion_app import _backend_http_exception
from sentimentizer.serve.diffusion_models import GenerateRequest


def test_krea_workflow_uses_native_convrot_nodes() -> None:
    workflow = build_krea_2_workflow(
        KREA_2_CONFIG,
        prompt="a red apple",
        steps=8,
        guidance_scale=1.0,
        width=1024,
        height=768,
        seed=42,
    )

    assert workflow["1"]["inputs"]["unet_name"] == "krea2_turbo_int8_convrot.safetensors"
    assert workflow["2"]["inputs"]["type"] == "krea2"
    assert workflow["7"]["inputs"]["positive"] == ["4", 0]
    assert workflow["7"]["inputs"]["negative"] == ["5", 0]
    assert workflow["preview_image"] == {
        "class_type": "PreviewImage",
        "inputs": {"images": ["8", 0]},
    }


def test_ideogram_workflow_uses_dual_models_and_native_scheduler() -> None:
    workflow = build_ideogram_4_workflow(
        IDEOGRAM_4_CONFIG,
        prompt='{"aspect_ratio":"1:1","high_level_description":"poster"}',
        steps=20,
        guidance_scale=7.0,
        width=1024,
        height=1024,
        seed=7,
    )

    assert workflow["2"]["inputs"]["unet_name"] == "ideogram4_int8_convrot.safetensors"
    assert (
        workflow["3"]["inputs"]["unet_name"] == "ideogram4_unconditional_int8_convrot.safetensors"
    )
    assert workflow["10"]["class_type"] == "Ideogram4Scheduler"
    assert workflow["11"]["inputs"] == {
        "model": ["2", 0],
        "cfg": 3.0,
        "start_percent": 0.7,
        "end_percent": 1.0,
    }
    assert workflow["12"]["inputs"]["model_negative"] == ["3", 0]


@pytest.mark.parametrize(
    ("steps", "expected"),
    [(8, (0.5, 1.75)), (20, (0.0, 1.75)), (48, (0.0, 1.5))],
)
def test_ideogram_schedule_presets(steps: int, expected: tuple[float, float]) -> None:
    assert _ideogram_schedule(steps) == expected


def test_client_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        ComfyUIClient("unix:///tmp/comfy.sock")


def test_ideogram_requires_license_acknowledgement() -> None:
    with pytest.raises(ValueError, match="license"):
        ServeConfig(ideogram_4_enabled=True, default_image_model="ideogram_4")


def test_krea_requires_license_acknowledgement() -> None:
    with pytest.raises(ValueError, match="license"):
        ServeConfig(
            krea_2_enabled=True,
            image_moderation_url="http://moderator.local/check",
            comfyui_temp_directory="/tmp/comfyui",
        )


def test_krea_requires_output_moderation() -> None:
    with pytest.raises(ValueError, match="image_moderation_url"):
        ServeConfig(
            krea_2_enabled=True,
            krea_2_license_accepted=True,
            comfyui_temp_directory="/tmp/comfyui",
        )


def test_enabled_image_model_requires_shared_temp_directory() -> None:
    with pytest.raises(ValueError, match="comfyui_temp_directory"):
        ServeConfig(
            ideogram_4_enabled=True,
            ideogram_4_license_accepted=True,
            default_image_model="ideogram_4",
        )


def test_default_image_model_must_be_enabled() -> None:
    with pytest.raises(ValueError, match="default_image_model"):
        ServeConfig(
            krea_2_enabled=True,
            krea_2_license_accepted=True,
            image_moderation_url="http://moderator.local/check",
            comfyui_temp_directory="/tmp/comfyui",
            default_image_model="ideogram_4",
        )


def test_ideogram_license_acknowledgement_allows_enablement() -> None:
    config = ServeConfig(
        ideogram_4_enabled=True,
        ideogram_4_license_accepted=True,
        comfyui_temp_directory="/tmp/comfyui",
        default_image_model="ideogram_4",
    )
    assert config.ideogram_4_enabled is True


def test_client_rejects_changed_prompt_id() -> None:
    client = ComfyUIClient("http://127.0.0.1:8188", poll_interval_s=0.001)
    history = {
        "prompt-1": {
            "outputs": {
                "preview_image": {
                    "images": [{"filename": "out.png", "subfolder": "", "type": "temp"}]
                }
            }
        }
    }

    def request_json(method: str, path: str, payload: object = None) -> dict[str, object]:
        if path == "/prompt":
            assert method == "POST"
            assert isinstance(payload, dict)
            assert payload["prompt_id"] == "00000000-0000-4000-8000-000000000001"
            return {"prompt_id": "prompt-1"}
        assert path == "/history/prompt-1"
        return history

    with (
        patch.object(client, "_request_json", side_effect=request_json),
        patch.object(client, "_request_bytes", return_value=b"png") as request_bytes,
        patch.object(client, "_delete_history") as delete_history,
        pytest.raises(ComfyUIError, match="changed prompt_id"),
    ):
        client.generate(
            "krea_2",
            KREA_2_CONFIG,
            prompt="apple",
            steps=8,
            guidance_scale=1.0,
            width=1024,
            height=1024,
            seed=42,
            prompt_id="00000000-0000-4000-8000-000000000001",
        )

    delete_history.assert_not_called()
    request_bytes.assert_not_called()


def test_client_queues_polls_and_fetches_matching_prompt() -> None:
    client = ComfyUIClient("http://127.0.0.1:8188", poll_interval_s=0.001)
    prompt_id = "00000000-0000-4000-8000-000000000001"
    history = {
        prompt_id: {
            "outputs": {
                "preview_image": {
                    "images": [{"filename": "out.png", "subfolder": "", "type": "temp"}]
                }
            }
        }
    }

    def request_json(method: str, path: str, payload: object = None) -> dict[str, object]:
        if path == "/prompt":
            return {"prompt_id": prompt_id}
        assert path == f"/history/{prompt_id}"
        return history

    with (
        patch.object(client, "_request_json", side_effect=request_json),
        patch.object(client, "_request_bytes", return_value=b"png") as request_bytes,
        patch.object(client, "_delete_history") as delete_history,
    ):
        result = client.generate(
            "krea_2",
            KREA_2_CONFIG,
            prompt="apple",
            steps=8,
            guidance_scale=1.0,
            width=1024,
            height=1024,
            seed=42,
            prompt_id=prompt_id,
        )

    assert result.data == b"png"
    assert result.seed == 42
    request_bytes.assert_called_once_with("GET", "/view?filename=out.png&subfolder=&type=temp")
    delete_history.assert_called_once_with(prompt_id)


def test_client_cancels_target_prompt() -> None:
    client = ComfyUIClient("http://127.0.0.1:8188")
    prompt_id = "00000000-0000-4000-8000-000000000001"
    with patch.object(
        client,
        "_request_json",
        return_value={"cancelled": True},
    ) as request_json:
        assert client.cancel(prompt_id) is True
    request_json.assert_called_once_with("POST", f"/api/jobs/{prompt_id}/cancel", {})


def test_client_removes_fetched_temporary_image(tmp_path) -> None:
    image_path = tmp_path / "out.png"
    image_path.write_bytes(b"png")
    client = ComfyUIClient("http://127.0.0.1:8188", temp_directory=tmp_path)

    client._remove_temp_image({"filename": "out.png", "subfolder": "", "type": "temp"})

    assert not image_path.exists()


def test_client_rejects_temp_path_escape(tmp_path) -> None:
    client = ComfyUIClient("http://127.0.0.1:8188", temp_directory=tmp_path)
    with pytest.raises(ComfyUIError, match="unsafe"):
        client._remove_temp_image(
            {"filename": "outside.png", "subfolder": "../outside", "type": "temp"}
        )


def test_client_does_not_queue_a_pre_cancelled_prompt() -> None:
    client = ComfyUIClient("http://127.0.0.1:8188")
    cancel_event = threading.Event()
    cancel_event.set()
    with (
        patch.object(client, "_request_json") as request_json,
        pytest.raises(ComfyUICancelled),
    ):
        client.generate(
            "krea_2",
            KREA_2_CONFIG,
            prompt="apple",
            steps=8,
            guidance_scale=1.0,
            width=1024,
            height=1024,
            seed=42,
            prompt_id="00000000-0000-4000-8000-000000000001",
            cancel_event=cancel_event,
        )
    request_json.assert_not_called()


def test_history_execution_error_is_reported() -> None:
    client = ComfyUIClient("http://127.0.0.1:8188", poll_interval_s=0.001)
    failed = {
        "prompt-1": {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [["execution_error", {"exception_message": "CUDA out of memory"}]],
            },
        }
    }
    with (
        patch.object(client, "_request_json", return_value=failed),
        pytest.raises(ComfyUIError, match="CUDA out of memory"),
    ):
        client._wait_for_image("prompt-1")


def test_seed_bounds_are_validated() -> None:
    assert GenerateRequest(prompt="apple", seed=2**64 - 1).seed == 2**64 - 1
    with pytest.raises(ValueError):
        GenerateRequest(prompt="apple", seed=-1)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="apple", seed=2**64)


def test_backend_timeout_maps_to_gateway_timeout() -> None:
    exc = _backend_http_exception(ComfyUIError("ComfyUI prompt timed out after 10s"))
    assert exc.status_code == 504
    assert exc.detail["code"] == "image_backend_timeout"


def test_backend_rejection_maps_to_bad_gateway_without_internal_detail() -> None:
    exc = _backend_http_exception(ComfyUIError("CUDA details that should not leak"))
    assert exc.status_code == 502
    assert exc.detail == {
        "code": "image_backend_error",
        "message": "Image backend rejected or failed the workflow",
    }


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_image_moderation_requires_explicit_safe_response() -> None:
    client = ImageModerationClient("http://moderator.local/check")
    with patch(
        "urllib.request.urlopen",
        return_value=_Response(b'{"safe": true}'),
    ) as urlopen:
        client.check(b"png", model="krea_2", user="user-1")

    request = urlopen.call_args.args[0]
    assert request.headers["Content-type"] == "application/json"
    assert b'"model": "krea_2"' in request.data


def test_image_moderation_blocks_unsafe_output() -> None:
    client = ImageModerationClient("http://moderator.local/check")
    with (
        patch(
            "urllib.request.urlopen",
            return_value=_Response(
                b'{"safe": false, "code": "unsafe_image", "message": "blocked"}'
            ),
        ),
        pytest.raises(UnsafeImageError, match="blocked"),
    ):
        client.check(b"png", model="krea_2", user=None)


def test_image_moderation_fails_closed_on_bad_schema() -> None:
    client = ImageModerationClient("http://moderator.local/check")
    with (
        patch("urllib.request.urlopen", return_value=_Response(b"{}")),
        pytest.raises(ImageModerationError, match="boolean 'safe'"),
    ):
        client.check(b"png", model="krea_2", user=None)
