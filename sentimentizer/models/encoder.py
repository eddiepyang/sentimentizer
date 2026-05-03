import math
from importlib.resources import files

import numpy as np
import torch
from gensim import corpora
from torch import nn

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, VALID_DEVICES, EmbeddingsConfig, EncoderConfig
from sentimentizer.extractor import new_embedding_weights

logger = new_logger(DEFAULT_LOG_LEVEL)

# Module-level singleton for B008 compliance — used as default in function signatures
_DEFAULT_ENCODER_CONFIG = EncoderConfig()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'.

    Injects position information into the input so the Transformer can
    distinguish word order — critical for sentiment (e.g. "not good" vs "good not").
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, seq_len, d_model) with position info added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """Transformer encoder for sentiment classification.

    Uses full self-attention over the token sequence, a learnable CLS token
    for sentence-level representation, sinusoidal positional encoding, and
    a multi-layer Transformer encoder with batch_first=True.

    Args:
        input_len: Maximum sequence length (number of tokens)
        d_model: Internal Transformer dimension (projected from embeddings)
        n_heads: Number of attention heads
        emb_weights: Pre-trained embedding weights of shape (vocab_size, emb_dim)
        n_layers: Number of Transformer encoder layers
        verbose: Whether to log debug shapes
        dropout: Dropout probability
        ff_multiplier: Feed-forward dim = d_model * ff_multiplier
    """

    def __init__(
        self,
        input_len: int,
        emb_weights: torch.Tensor,
        d_model: int = EncoderConfig.d_model,
        n_heads: int = EncoderConfig.n_heads,
        n_layers: int = EncoderConfig.n_layers,
        verbose: bool = True,
        dropout: float = EncoderConfig.dropout,
        ff_multiplier: int = EncoderConfig.ff_multiplier,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Embedding layer (vocab_size, emb_dim)
        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1])

        # Project GloVe embeddings to d_model dimension
        self.proj = nn.Linear(emb_weights.shape[1], d_model)

        # Learnable CLS token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding (max_len = input_len + 1 for CLS token)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=input_len + 1)

        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # Classification head: CLS token → logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Load embedding weights immediately
        self.embed_layer.load_state_dict({"weight": emb_weights})  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            inputs: Token IDs of shape (batch, seq_len)
                    Zero-padding is used to build the padding mask.

        Returns:
            Logits of shape (batch,)
        """
        embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
        logger.debug(f"embedding shape {embeds.shape}")

        # Project to d_model
        projected = self.proj(embeds)  # (B, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(inputs.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, projected], dim=1)  # (B, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, seq_len+1, d_model)

        # Padding mask: True where padding (so attention ignores those positions).
        # CLS token at position 0 is never padded, so prepend False.
        pad_mask = inputs == 0  # (B, seq_len)
        cls_mask = torch.zeros(inputs.size(0), 1, dtype=torch.bool, device=inputs.device)
        src_key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, seq_len+1)

        # Transformer encoder — self-attention with padding mask
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, seq_len+1, d_model)
        logger.debug(f"encoder out shape {encoded.shape}")

        # Pool from CLS token position
        cls_out = encoded[:, 0, :]  # (B, d_model)
        logger.debug(f"cls out shape {cls_out.shape}")

        # Classify
        logits = self.classifier(cls_out)  # (B, 1)
        return torch.squeeze(logits)  # (B,)

    def predict(self, converted_text: np.ndarray) -> torch.Tensor:
        """Run inference with sigmoid activation.

        Args:
            converted_text: Token IDs as numpy array

        Returns:
            Sentiment score between 0 (negative) and 1 (positive)
        """
        with torch.no_grad():
            self.eval()
            output = torch.from_numpy(converted_text)
            return torch.sigmoid(self.forward(output))


def new_model(
    dict_path: str,
    embeddings_config: EmbeddingsConfig,
    input_len: int,
    model_config: EncoderConfig = _DEFAULT_ENCODER_CONFIG,
) -> Encoder:
    """Create a new Encoder model with pre-trained GloVe embeddings.

    Args:
        dict_path: Path to the gensim dictionary file
        embeddings_config: Configuration for GloVe embeddings
        input_len: Maximum sequence length
        model_config: Encoder architecture configuration (defaults from EncoderConfig)
    """
    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = Encoder(
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        input_len=input_len,
        emb_weights=emb_t,
        n_layers=model_config.n_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
    )
    return model


def get_trained_model(
    device: str,
    model_config: EncoderConfig = _DEFAULT_ENCODER_CONFIG,
) -> Encoder:
    """Load a pre-trained Encoder model from saved weights.

    Args:
        device: Device to load weights onto ("cpu", "cuda", or "mps")
        model_config: Encoder architecture configuration (must match saved weights)

    Returns:
        Encoder model with loaded weights
    """
    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    weights_path = str(files("sentimentizer.data").joinpath("encoder_weights.pth"))

    try:
        weights = torch.load(
            weights_path,
            map_location=torch.device(device=device),
            weights_only=True,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Weights file not found at {weights_path}. "
            "Please train the model first: "
            "python workflows/driver.py --device cuda --model encoder --type new --save True"
        ) from e

    # Infer vocab size and d_model from saved weights
    emb_shape = weights["embed_layer.weight"].shape
    d_model = weights["proj.weight"].shape[0]  # output dim of proj layer

    empty_embeddings = torch.zeros(emb_shape)
    model = Encoder(
        d_model=d_model,
        n_heads=model_config.n_heads,
        input_len=200,
        emb_weights=empty_embeddings,
        n_layers=model_config.n_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
    )

    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        raise RuntimeError(
            "Saved weights are incompatible with the current model architecture. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model encoder --type new --save True"
        ) from e

    return model
