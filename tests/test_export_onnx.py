"""Tests for ONNX export, quantization, and validation.

Tests use pytest.mark.skipif to skip when onnxruntime is not available,
allowing CI to run without the onnx dependency.
"""

import pytest

torch = pytest.importorskip("torch")

# Check if onnxruntime is available
try:
    import onnxruntime  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

skip_without_onnx = pytest.mark.skipif(
    not ONNX_AVAILABLE,
    reason="onnxruntime not installed (install with: pip install -e '.[onnx]')",
)


class TestRNNOnnxExportMode:
    """Test that RNN's onnx_export flag produces valid but different outputs."""

    def test_onnx_export_flag_changes_forward_path(self) -> None:
        """onnx_export=True should produce different output than onnx_export=False
        due to the masked fallback path."""
        from sentimentizer.models.rnn import RNN

        # Create a small RNN model
        emb_weights = torch.randn(100, 50)  # vocab_size=100, emb_dim=50
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()

        # Create input with some padding (zeros)
        inputs = torch.zeros(2, 10, dtype=torch.long)
        inputs[0, :7] = torch.randint(1, 99, (7,))  # 7 real tokens
        inputs[1, :4] = torch.randint(1, 99, (4,))  # 4 real tokens

        with torch.no_grad():
            output_packed = model(inputs, onnx_export=False)
            output_onnx = model(inputs, onnx_export=True)

        # Both should produce valid outputs (batch_size, num_classes=3)
        assert output_packed.shape == (2, 3)
        assert output_onnx.shape == (2, 3)

        # Outputs should be different due to different forward paths
        # (but not too different — same model, same weights)
        assert not torch.allclose(output_packed, output_onnx, atol=1e-6)

    def test_onnx_export_default_is_false(self) -> None:
        """Default onnx_export should be False (standard packed path)."""
        from sentimentizer.models.rnn import RNN

        emb_weights = torch.randn(100, 50)
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()

        inputs = torch.randint(1, 99, (2, 10), dtype=torch.long)

        with torch.no_grad():
            output_default = model(inputs)  # no onnx_export arg
            output_explicit_false = model(inputs, onnx_export=False)

        assert torch.allclose(output_default, output_explicit_false, atol=1e-6)


class TestRNNOnnxWrapper:
    """Test that _RNNOnnxWrapper correctly calls forward with onnx_export=True."""

    def test_wrapper_calls_onnx_export_true(self) -> None:
        """_RNNOnnxWrapper should produce same output as RNN(onnx_export=True)."""
        from sentimentizer.export_onnx import _RNNOnnxWrapper
        from sentimentizer.models.rnn import RNN

        emb_weights = torch.randn(100, 50)
        rnn = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        rnn.eval()
        wrapper = _RNNOnnxWrapper(rnn)
        wrapper.eval()

        inputs = torch.zeros(2, 10, dtype=torch.long)
        inputs[0, :7] = torch.randint(1, 99, (7,))
        inputs[1, :4] = torch.randint(1, 99, (4,))

        with torch.no_grad():
            wrapper_output = wrapper(inputs)
            rnn_onnx_output = rnn(inputs, onnx_export=True)

        assert torch.allclose(wrapper_output, rnn_onnx_output, atol=1e-6)


@skip_without_onnx
class TestExportPipeline:
    """Test ONNX export pipeline (requires onnxruntime)."""

    @pytest.fixture
    def small_rnn(self):
        """Create a small RNN model for testing."""
        from sentimentizer.models.rnn import RNN

        emb_weights = torch.randn(100, 50)
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()
        return model

    @pytest.fixture
    def small_encoder(self):
        """Create a small Encoder model for testing."""
        from sentimentizer.models.encoder import Encoder

        emb_weights = torch.randn(100, 50)
        model = Encoder(emb_weights=emb_weights, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
        model.eval()
        return model

    def test_export_rnn_to_onnx(self, small_rnn, tmp_path) -> None:
        """Export RNN to ONNX and verify the file exists."""
        from sentimentizer.export_onnx import export_model_to_onnx

        output_path = tmp_path / "rnn_test.onnx"
        result = export_model_to_onnx(small_rnn, "rnn", output_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_encoder_to_onnx(self, small_encoder, tmp_path) -> None:
        """Export Encoder to ONNX and verify the file exists."""
        from sentimentizer.export_onnx import export_model_to_onnx

        output_path = tmp_path / "encoder_test.onnx"
        result = export_model_to_onnx(small_encoder, "encoder", output_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_validate_onnx_export_rnn(self, small_rnn, tmp_path) -> None:
        """Validate RNN ONNX export matches PyTorch output within tolerance."""
        from sentimentizer.export_onnx import export_model_to_onnx, validate_onnx_export

        output_path = tmp_path / "rnn_validate.onnx"
        export_model_to_onnx(small_rnn, "rnn", output_path)

        test_input = torch.randint(1, 99, (4, 20), dtype=torch.long)
        result = validate_onnx_export(output_path, small_rnn, "rnn", test_input, tolerance=1e-2)
        assert result["passed"], f"Validation failed: max_diff={result['max_diff']}"
        assert result["max_diff"] < 1e-2

    def test_validate_onnx_export_encoder(self, small_encoder, tmp_path) -> None:
        """Validate Encoder ONNX export matches PyTorch output within tolerance."""
        from sentimentizer.export_onnx import export_model_to_onnx, validate_onnx_export

        output_path = tmp_path / "encoder_validate.onnx"
        export_model_to_onnx(small_encoder, "encoder", output_path)

        test_input = torch.randint(1, 99, (4, 20), dtype=torch.long)
        result = validate_onnx_export(
            output_path, small_encoder, "encoder", test_input, tolerance=1e-4
        )
        assert result["passed"], f"Validation failed: max_diff={result['max_diff']}"
        assert result["max_diff"] < 1e-4

    def test_quantize_onnx_model(self, small_rnn, tmp_path) -> None:
        """Test INT8 quantization of ONNX model."""
        from sentimentizer.export_onnx import export_model_to_onnx, quantize_onnx_model

        onnx_path = tmp_path / "rnn_quantize_test.onnx"
        export_model_to_onnx(small_rnn, "rnn", onnx_path)

        quantized_path = quantize_onnx_model(onnx_path)
        assert quantized_path.exists()
        assert quantized_path.stat().st_size > 0
        # Quantized model should be smaller than FP32
        assert quantized_path.stat().st_size < onnx_path.stat().st_size

    def test_rnn_tolerance_relaxed(self, small_rnn, tmp_path) -> None:
        """RNN should use tolerance 1e-2 (relaxed) by default."""
        from sentimentizer.export_onnx import export_model_to_onnx, validate_onnx_export

        output_path = tmp_path / "rnn_tolerance.onnx"
        export_model_to_onnx(small_rnn, "rnn", output_path)

        test_input = torch.randint(1, 99, (4, 20), dtype=torch.long)
        result = validate_onnx_export(output_path, small_rnn, "rnn", test_input)
        # Default tolerance for RNN should be 1e-2
        assert result["tolerance"] == 1e-2

    def test_encoder_tolerance_strict(self, small_encoder, tmp_path) -> None:
        """Encoder should use tolerance 1e-4 (strict) by default."""
        from sentimentizer.export_onnx import export_model_to_onnx, validate_onnx_export

        output_path = tmp_path / "encoder_tolerance.onnx"
        export_model_to_onnx(small_encoder, "encoder", output_path)

        test_input = torch.randint(1, 99, (4, 20), dtype=torch.long)
        result = validate_onnx_export(output_path, small_encoder, "encoder", test_input)
        # Default tolerance for Encoder should be 1e-4
        assert result["tolerance"] == 1e-4
