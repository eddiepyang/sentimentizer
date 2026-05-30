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
        with (
            patch("sentimentizer.diffusion.config.is_mlx_device", return_value=True),
            patch.dict(BACKEND_REGISTRY, {"flux2_klein": ["diffusers", "mlx"]}),
        ):
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

    def test_empty_string_quantize_maps_to_none(self) -> None:
        from sentimentizer.diffusion.mlx_predictor import _MFLUX_QUANTIZE_MAP

        assert _MFLUX_QUANTIZE_MAP[""] is None


class TestDiffusionPredictorProtocol:
    def test_diffusers_predictor_satisfies_protocol(self) -> None:
        from sentimentizer.diffusion.predictor import DiffusionPredictorProtocol

        p = create_predictor("sd35")
        assert isinstance(p, DiffusionPredictorProtocol)

    def test_flux2_klein_predictor_satisfies_protocol(self) -> None:
        from sentimentizer.diffusion.predictor import DiffusionPredictorProtocol

        cfg = DiffusionModelConfig(backend="diffusers")
        p = create_predictor("flux2_klein", cfg)
        assert isinstance(p, DiffusionPredictorProtocol)


class TestCreatePredictorMlxBackend:
    def test_explicit_mlx_without_mflux_raises_value_error(self) -> None:
        if _MFLUX_INSTALLED:
            pytest.skip("mflux is installed; cannot test no-mflux path")
        cfg = DiffusionModelConfig(backend="mlx")
        with pytest.raises(ValueError, match="not available"):
            create_predictor("flux2_klein", cfg)

    def test_explicit_mlx_with_mflux_but_not_mlx_device_raises_import_error(
        self,
    ) -> None:
        if not _MFLUX_INSTALLED:
            pytest.skip("mflux not installed")
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        registry = {
            "flux2_klein": {
                "diffusers": Flux2KleinPredictor,
                "mlx": MLXFlux2KleinPredictor,
            },
        }
        with (
            patch.dict(BACKEND_REGISTRY, {"flux2_klein": ["diffusers", "mlx"]}),
            patch.dict(
                "sentimentizer.diffusion.predictor._PREDICTOR_REGISTRY",
                registry,
            ),
            patch("sentimentizer.diffusion.config.is_mlx_device", return_value=False),
        ):
            cfg = DiffusionModelConfig(backend="mlx")
            with pytest.raises(ValueError, match="not available"):
                create_predictor("flux2_klein", cfg)

    def test_cpu_offload_warning_for_mlx(self) -> None:
        if not _MFLUX_INSTALLED:
            pytest.skip("mflux not installed")
        from sentimentizer.diffusion.config import resolve_backend
        from sentimentizer.diffusion.mlx_predictor import MLXFlux2KleinPredictor

        registry = {
            "flux2_klein": {
                "diffusers": Flux2KleinPredictor,
                "mlx": MLXFlux2KleinPredictor,
            },
        }
        with (
            patch.dict(BACKEND_REGISTRY, {"flux2_klein": ["diffusers", "mlx"]}),
            patch.dict(
                "sentimentizer.diffusion.predictor._PREDICTOR_REGISTRY",
                registry,
            ),
            patch("sentimentizer.diffusion.config.is_mlx_device", return_value=True),
        ):
            backend = resolve_backend("flux2_klein", "mlx")
            assert backend == "mlx"


class TestBasePredictorModelInfo:
    def test_sdxl_includes_backend_diffusers(self) -> None:
        p = SDXLPredictor()
        info = p.model_info()
        assert info["backend"] == "diffusers"

    def test_sd35_includes_backend_diffusers(self) -> None:
        p = SD35Predictor()
        info = p.model_info()
        assert info["backend"] == "diffusers"

    def test_flux2_klein_includes_backend_diffusers(self) -> None:
        p = Flux2KleinPredictor()
        info = p.model_info()
        assert info["backend"] == "diffusers"
