"""Characterization tests for the predict_text refactoring.

These tests verify that the combined predict_text() method on BaseSentimentModel
produces identical results to the previous two-step pattern of
tokenizer.tokenize_text() → model.predict() → .item().
"""

import pytest
import torch

from sentimentizer.models.base import BaseSentimentModel
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.rnn import RNN
from sentimentizer.tokenizer import Tokenizer

# ─── Helpers ────────────────────────────────────────────────────


def _make_tokenizer(tmp_path):
    """Create a Tokenizer with a small dictionary for testing."""
    from gensim import corpora

    words = ["good", "bad", "great", "terrible", "the", "a", "is", "not"]
    dictionary = corpora.Dictionary([words])
    tokenizer = Tokenizer(dictionary=dictionary)
    return tokenizer


def _make_rnn_model():
    """Create a small RNN model for testing."""
    vocab_size = 50
    emb_dim = 10
    hidden_size = 20
    emb_weights = torch.randn(vocab_size, emb_dim)
    model = RNN(emb_weights=emb_weights, hidden_size=hidden_size, num_layers=1, dropout=0.0)
    model.eval()
    return model


def _make_encoder_model():
    """Create a small Encoder model for testing."""
    vocab_size = 50
    emb_dim = 10
    d_model = 16
    emb_weights = torch.randn(vocab_size, emb_dim)
    model = Encoder(
        emb_weights=emb_weights,
        d_model=d_model,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model.eval()
    return model


def _make_decoder_model():
    """Create a small Decoder model for testing."""
    vocab_size = 50
    emb_dim = 10
    d_model = 16
    emb_weights = torch.randn(vocab_size, emb_dim)
    model = Decoder(
        emb_weights=emb_weights,
        d_model=d_model,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    model.eval()
    return model


# ─── BaseSentimentModel ─────────────────────────────────────────


class TestBaseSentimentModel:
    """Test that BaseSentimentModel is properly inherited by all model classes."""

    def test_rnn_inherits_from_base(self):
        assert issubclass(RNN, BaseSentimentModel)

    def test_encoder_inherits_from_base(self):
        assert issubclass(Encoder, BaseSentimentModel)

    def test_decoder_inherits_from_base(self):
        assert issubclass(Decoder, BaseSentimentModel)

    def test_rnn_instance_is_base_sentiment_model(self):
        model = _make_rnn_model()
        assert isinstance(model, BaseSentimentModel)

    def test_encoder_instance_is_base_sentiment_model(self):
        model = _make_encoder_model()
        assert isinstance(model, BaseSentimentModel)

    def test_decoder_instance_is_base_sentiment_model(self):
        model = _make_decoder_model()
        assert isinstance(model, BaseSentimentModel)

    def test_predict_text_method_exists(self):
        """Verify predict_text is defined on BaseSentimentModel."""
        assert hasattr(BaseSentimentModel, "predict_text")
        assert callable(BaseSentimentModel.predict_text)

    def test_predict_method_exists(self):
        """Verify predict is defined on BaseSentimentModel."""
        assert hasattr(BaseSentimentModel, "predict")
        assert callable(BaseSentimentModel.predict)


# ─── predict_text characterization ──────────────────────────────


class TestPredictTextCharacterization:
    """Verify predict_text() matches the two-step pattern exactly."""

    @pytest.fixture
    def tokenizer(self, tmp_path):
        return _make_tokenizer(tmp_path)

    def test_predict_text_matches_two_step_rnn(self, tokenizer):
        """predict_text should give same result as tokenize → predict → item for RNN."""
        model = _make_rnn_model()
        text = "the good great"

        # Two-step pattern
        token_ids = tokenizer.tokenize_text(text)
        two_step_score = model.predict(token_ids).item()

        # Combined method
        combined_score = model.predict_text(text, tokenizer)

        assert combined_score == pytest.approx(two_step_score)

    def test_predict_text_matches_two_step_encoder(self, tokenizer):
        """predict_text should give same result as tokenize → predict → item for Encoder."""
        model = _make_encoder_model()
        text = "the good great"

        # Two-step pattern
        token_ids = tokenizer.tokenize_text(text)
        two_step_score = model.predict(token_ids).item()

        # Combined method
        combined_score = model.predict_text(text, tokenizer)

        assert combined_score == pytest.approx(two_step_score)

    def test_predict_text_matches_two_step_decoder(self, tokenizer):
        """predict_text should give same result as tokenize → predict → item for Decoder."""
        model = _make_decoder_model()
        text = "the good great"

        # Two-step pattern
        token_ids = tokenizer.tokenize_text(text)
        two_step_score = model.predict(token_ids).item()

        # Combined method
        combined_score = model.predict_text(text, tokenizer)

        assert combined_score == pytest.approx(two_step_score)

    def test_predict_text_returns_float(self, tokenizer):
        """predict_text should return a Python float, not a tensor."""
        model = _make_rnn_model()
        score = model.predict_text("the good great", tokenizer)
        assert isinstance(score, float)

    def test_predict_text_score_in_range(self, tokenizer):
        """predict_text score should be between 0 and 1 (sigmoid output)."""
        model = _make_rnn_model()
        score = model.predict_text("the good great", tokenizer)
        assert 0.0 <= score <= 1.0

    def test_predict_text_no_gradient(self, tokenizer):
        """predict_text should not compute gradients."""
        model = _make_rnn_model()
        with torch.no_grad():
            # Calling predict_text inside no_grad should work fine
            score = model.predict_text("the good great", tokenizer)
        assert isinstance(score, float)
