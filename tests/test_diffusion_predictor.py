"""Tests for diffusion predictor module (CPU-only, using tiny-SD model)."""

from __future__ import annotations

import pytest

from sentimentizer.diffusion.config import (
    FLUX_DEFAULT_CONFIG,
    SD_DEFAULT_CONFIG,
    DiffusionModelConfig,
)
from sentimentizer.diffusion.predictor import (
    FluxPredictor,
    SD35Predictor,
    SDPredictor,
    _b64,
    _encode_pil,
    _generate_id,
    _resolve_dtype,
)


class TestResolveDtype:
    def test_bfloat16(self) -> None:
        import torch

        assert _resolve_dtype("bfloat16") == torch.bfloat16

    def test_float16(self) -> None:
        import torch

        assert _resolve_dtype("float16") == torch.float16

    def test_float32(self) -> None:
        import torch

        assert _resolve_dtype("float32") == torch.float32

    def test_unknown_defaults_bfloat16(self) -> None:
        import torch

        assert _resolve_dtype("unknown") == torch.bfloat16


class TestGenerateId:
    def test_prefix(self) -> None:
        id_ = _generate_id()
        assert id_.startswith("img_")

    def test_length(self) -> None:
        id_ = _generate_id()
        assert len(id_) == 16  # "img_" + 12 chars

    def test_uniqueness(self) -> None:
        ids = {_generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestEncodePil:
    def test_png(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        data = _encode_pil(img, "png")
        assert data[:4] == b"\x89PNG"

    def test_jpeg(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        data = _encode_pil(img, "jpeg")
        assert data[:2] == b"\xff\xd8"

    def test_webp(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        data = _encode_pil(img, "webp")
        assert data[:4] == b"RIFF"


class TestB64:
    def test_roundtrip(self) -> None:
        import base64

        raw = b"hello world"
        encoded = _b64(raw)
        assert base64.b64decode(encoded) == raw


class TestDiffusionModelConfig:
    def test_sd_defaults(self) -> None:
        assert SD_DEFAULT_CONFIG.default_steps == 30
        assert SD_DEFAULT_CONFIG.default_guidance == 7.5
        assert SD_DEFAULT_CONFIG.dim_alignment == 8

    def test_flux_defaults(self) -> None:
        assert FLUX_DEFAULT_CONFIG.default_steps == 28
        assert FLUX_DEFAULT_CONFIG.default_guidance == 3.5
        assert FLUX_DEFAULT_CONFIG.dim_alignment == 16

    def test_frozen(self) -> None:
        cfg = DiffusionModelConfig()
        with pytest.raises(AttributeError):
            cfg.default_steps = 99  # type: ignore[misc]


class TestSDPredictorLifecycle:
    def test_init_not_loaded(self) -> None:
        p = SDPredictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_resolve_defaults_with_values(self) -> None:
        from types import SimpleNamespace

        p = SDPredictor()
        req = SimpleNamespace(
            prompt="test",
            negative_prompt=None,
            steps=None,
            guidance_scale=None,
            width=512,
            height=512,
            seed=42,
            output_format="png",
            response_format="b64_json",
        )
        resolved = p.resolve_defaults(req)
        assert resolved["steps"] == 30
        assert resolved["guidance_scale"] == 7.5
        assert resolved["seed"] == 42

    def test_resolve_defaults_explicit(self) -> None:
        from types import SimpleNamespace

        p = SDPredictor()
        req = SimpleNamespace(
            prompt="test",
            negative_prompt="blurry",
            steps=20,
            guidance_scale=5.0,
            width=768,
            height=768,
            seed=None,
            output_format="webp",
            response_format="url",
        )
        resolved = p.resolve_defaults(req)
        assert resolved["steps"] == 20
        assert resolved["guidance_scale"] == 5.0
        assert resolved["negative_prompt"] == "blurry"

    def test_model_info_not_loaded(self) -> None:
        p = SDPredictor()
        info = p.model_info()
        assert info["status"] == "not_loaded"
        assert info["default_steps"] == 30

    def test_generate_raises_if_not_loaded(self) -> None:
        p = SDPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")


class TestFluxPredictorDefaults:
    def test_init_not_loaded(self) -> None:
        p = FluxPredictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_flux_default_guidance(self) -> None:
        assert FluxPredictor().cfg.default_guidance == 3.5

    def test_generate_raises_if_not_loaded(self) -> None:
        p = FluxPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")


class TestSD35PredictorDefaults:
    def test_init_not_loaded(self) -> None:
        p = SD35Predictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_sd35_default_guidance(self) -> None:
        assert SD35Predictor().cfg.default_guidance == 4.5

    def test_sd35_default_steps(self) -> None:
        assert SD35Predictor().cfg.default_steps == 40

    def test_sd35_default_model_id(self) -> None:
        assert SD35Predictor().cfg.model_id == "stabilityai/stable-diffusion-3.5-medium"

    def test_sd35_dim_alignment(self) -> None:
        assert SD35Predictor().cfg.dim_alignment == 16

    def test_generate_raises_if_not_loaded(self) -> None:
        p = SD35Predictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")


class TestSeedResolution:
    def test_explicit_seed(self) -> None:
        p = SDPredictor()
        assert p._resolve_seed(42) == 42

    def test_random_seed(self) -> None:
        p = SDPredictor()
        s = p._resolve_seed(None)
        assert 0 <= s <= 2**32 - 1

    def test_invalid_seed(self) -> None:
        p = SDPredictor()
        with pytest.raises(ValueError):
            p._resolve_seed(-1)
        with pytest.raises(ValueError):
            p._resolve_seed(2**32)
