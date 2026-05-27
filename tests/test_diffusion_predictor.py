"""Tests for diffusion predictor module (CPU-only)."""

from __future__ import annotations

import pytest

from sentimentizer.diffusion.config import (
    FLUX2_KLEIN_DEFAULT_CONFIG,
    SD35_DEFAULT_CONFIG,
    SDXL_DEFAULT_CONFIG,
    DiffusionModelConfig,
)
from sentimentizer.diffusion.predictor import (
    Flux2KleinPredictor,
    SD35Predictor,
    SDXLPredictor,
    _b64,
    _encode_pil,
    _generate_id,
    _resolve_dtype,
)

_PIL_AVAILABLE: bool = False
try:
    import PIL  # noqa: F401

    _PIL_AVAILABLE = True
except ModuleNotFoundError:
    pass


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
    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow not installed")
    def test_png(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        data = _encode_pil(img, "png")
        assert data[:4] == b"\x89PNG"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow not installed")
    def test_jpeg(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="red")
        data = _encode_pil(img, "jpeg")
        assert data[:2] == b"\xff\xd8"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="pillow not installed")
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


class TestDecodeB64Image:
    def test_raw_b64(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        decoded = _decode_b64_image(b64, max_pixels=64 * 64)
        assert decoded.mode == "RGB"
        assert decoded.size == (64, 64)

    def test_data_url(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        decoded = _decode_b64_image(b64, max_pixels=64 * 64)
        assert decoded.mode == "RGB"
        assert decoded.size == (64, 64)

    def test_rgba_to_rgb(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        decoded = _decode_b64_image(b64, max_pixels=64 * 64)
        assert decoded.mode == "RGB"

    def test_malformed_b64(self) -> None:
        from sentimentizer.diffusion.predictor import _decode_b64_image

        with pytest.raises(ValueError, match="malformed base64"):
            _decode_b64_image("not-base64", max_pixels=64 * 64)

    def test_exceeds_max_pixels(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGB", (128, 128), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        with pytest.raises(ValueError, match="exceeds max_pixels="):
            _decode_b64_image(b64, max_pixels=64 * 64)

    def test_exact_max_pixels(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        decoded = _decode_b64_image(b64, max_pixels=64 * 64)
        assert decoded.size == (64, 64)

    def test_non_square(self) -> None:
        import base64
        import io

        import PIL.Image

        from sentimentizer.diffusion.predictor import _decode_b64_image

        img = PIL.Image.new("RGB", (128, 32), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        decoded = _decode_b64_image(b64, max_pixels=64 * 64)
        assert decoded.size == (128, 32)


class TestDiffusionModelConfig:
    def test_sd35_defaults(self) -> None:
        assert SD35_DEFAULT_CONFIG.default_steps == 40
        assert SD35_DEFAULT_CONFIG.default_guidance == 4.5
        assert SD35_DEFAULT_CONFIG.dim_alignment == 16

    def test_sdxl_defaults(self) -> None:
        assert SDXL_DEFAULT_CONFIG.default_steps == 25
        assert SDXL_DEFAULT_CONFIG.default_guidance == 5.0
        assert SDXL_DEFAULT_CONFIG.dim_alignment == 8

    def test_flux2_klein_defaults(self) -> None:
        assert FLUX2_KLEIN_DEFAULT_CONFIG.default_steps == 4
        assert FLUX2_KLEIN_DEFAULT_CONFIG.default_guidance == 0.0
        assert FLUX2_KLEIN_DEFAULT_CONFIG.dim_alignment == 16

    def test_frozen(self) -> None:
        cfg = DiffusionModelConfig()
        with pytest.raises(AttributeError):
            cfg.default_steps = 99  # type: ignore[misc]


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


class TestFlux2KleinPredictorDefaults:
    def test_init_not_loaded(self) -> None:
        p = Flux2KleinPredictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_default_steps(self) -> None:
        assert Flux2KleinPredictor().cfg.default_steps == 4

    def test_default_guidance_unguided(self) -> None:
        assert Flux2KleinPredictor().cfg.default_guidance == 0.0

    def test_default_model_id(self) -> None:
        assert Flux2KleinPredictor().cfg.model_id == "black-forest-labs/FLUX.2-klein-4B"

    def test_dim_alignment(self) -> None:
        assert Flux2KleinPredictor().cfg.dim_alignment == 16

    def test_generate_raises_if_not_loaded(self) -> None:
        p = Flux2KleinPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")


class TestSDXLPredictorDefaults:
    def test_init_not_loaded(self) -> None:
        p = SDXLPredictor()
        assert not p.model_loaded
        assert p.model_error is None

    def test_generate_raises_if_not_loaded(self) -> None:
        p = SDXLPredictor()
        with pytest.raises(RuntimeError, match="not loaded"):
            p.generate("test prompt")


class TestResolveDefaults:
    def test_resolve_defaults_with_unset_request_fields(self) -> None:
        from types import SimpleNamespace

        p = SD35Predictor()
        req = SimpleNamespace(
            prompt="test",
            negative_prompt=None,
            steps=None,
            guidance_scale=None,
            width=1024,
            height=1024,
            seed=42,
            output_format="png",
            response_format="b64_json",
        )
        resolved = p.resolve_defaults(req)
        assert resolved["steps"] == 40
        assert resolved["guidance_scale"] == 4.5
        assert resolved["seed"] == 42

    def test_resolve_defaults_with_explicit_request_fields(self) -> None:
        from types import SimpleNamespace

        p = SD35Predictor()
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
        p = SD35Predictor()
        info = p.model_info()
        assert info["status"] == "not_loaded"
        assert info["default_steps"] == 40


class TestSeedResolution:
    def test_explicit_seed(self) -> None:
        p = SD35Predictor()
        assert p._resolve_seed(42) == 42

    def test_random_seed(self) -> None:
        p = SD35Predictor()
        s = p._resolve_seed(None)
        assert 0 <= s <= 2**32 - 1

    def test_invalid_seed(self) -> None:
        p = SD35Predictor()
        with pytest.raises(ValueError):
            p._resolve_seed(-1)
        with pytest.raises(ValueError):
            p._resolve_seed(2**32)


class TestReferenceImagesPredictorSupport:
    def test_sdxl_not_implemented(self) -> None:
        from unittest.mock import MagicMock

        p = SDXLPredictor()
        with pytest.raises(NotImplementedError):
            p.generate("test prompt", reference_images=[MagicMock()])

    def test_sd35_not_implemented(self) -> None:
        from unittest.mock import MagicMock

        p = SD35Predictor()
        with pytest.raises(NotImplementedError):
            p.generate("test prompt", reference_images=[MagicMock()])
