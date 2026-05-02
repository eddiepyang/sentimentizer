import math
from importlib.resources import files

import numpy as np
import torch
from gensim import corpora
from torch import nn

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, DecoderConfig, Devices, EmbeddingsConfig
from sentimentizer.extractor import new_embedding_weights

logger = new_logger(DEFAULT_LOG_LEVEL)

# Module-level singleton for B008 compliance — used as default in function signatures
_DEFAULT_DECODER_CONFIG = DecoderConfig()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe: torch.Tensor = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Decoder(nn.Module):
    """Encoder-Decoder Transformer for sentiment classification.

    The input text is encoded by a small TransformerEncoder into memory
    representations. A learnable query token cross-attends to the memory
    through TransformerDecoder layers. The query output is then classified
    into a sentiment score.

    Args:
        batch_size: Batch size
        input_len: Maximum sequence length (number of tokens)
        emb_weights: Pre-trained embedding weights of shape (vocab_size, emb_dim)
        d_model: Internal Transformer dimension (projected from embeddings)
        n_heads: Number of attention heads
        n_encoder_layers: Number of Transformer encoder layers
        n_decoder_layers: Number of Transformer decoder layers
        verbose: Whether to log debug shapes
        dropout: Dropout probability
        ff_multiplier: Feed-forward dim = d_model * ff_multiplier
    """

    def __init__(
        self,
        batch_size: int,
        input_len: int,
        emb_weights: torch.Tensor,
        d_model: int = DecoderConfig.d_model,
        n_heads: int = DecoderConfig.n_heads,
        n_encoder_layers: int = DecoderConfig.n_encoder_layers,
        n_decoder_layers: int = DecoderConfig.n_decoder_layers,
        verbose: bool = True,
        dropout: float = DecoderConfig.dropout,
        ff_multiplier: int = DecoderConfig.ff_multiplier,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.emb_weights = emb_weights
        self.d_model = d_model

        # Embedding layer
        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1])

        # Project GloVe embeddings to d_model
        self.proj = nn.Linear(emb_weights.shape[1], d_model)

        # Learnable query token for cross-attention decoding
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding for the input sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=input_len + 1)

        # Transformer encoder — encodes input text into memory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )

        # Transformer decoder — query token cross-attends to encoded memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
        )

        # Classification head: query output → logit
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.verbose = verbose

    def load_weights(self) -> "Decoder":
        """Load pre-trained GloVe embeddings into the embedding layer."""
        self.embed_layer.load_state_dict({"weight": self.emb_weights})  # type: ignore
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            inputs: Token IDs of shape (batch, seq_len)

        Returns:
            Logits of shape (batch,)
        """
        # Embed and project
        embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
        logger.debug(f"embedding shape {embeds.shape}")

        projected = self.proj(embeds)  # (B, seq_len, d_model)

        # Add positional encoding to input
        memory_input = self.pos_encoder(projected)  # (B, seq_len, d_model)

        # Encode input text into memory
        memory = self.encoder(memory_input)  # (B, seq_len, d_model)
        logger.debug(f"memory shape {memory.shape}")

        # Expand query token for the batch
        query = self.query_token.expand(inputs.size(0), -1, -1)  # (B, 1, d_model)

        # Query cross-attends to memory via decoder
        decoded = self.decoder(tgt=query, memory=memory)  # (B, 1, d_model)
        logger.debug(f"decoded shape {decoded.shape}")

        # Extract query output and classify
        query_out = decoded.squeeze(1)  # (B, d_model)
        logits = self.classifier(query_out)  # (B, 1)
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
    batch_size: int,
    input_len: int,
    model_config: DecoderConfig = _DEFAULT_DECODER_CONFIG,
) -> Decoder:
    """Create a new Decoder model with pre-trained GloVe embeddings.

    Args:
        dict_path: Path to the gensim dictionary file
        embeddings_config: Configuration for GloVe embeddings
        batch_size: Batch size
        input_len: Maximum sequence length
        model_config: Decoder architecture configuration (defaults from DecoderConfig)
    """
    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = Decoder(
        batch_size=batch_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        input_len=input_len,
        emb_weights=emb_t,
        n_encoder_layers=model_config.n_encoder_layers,
        n_decoder_layers=model_config.n_decoder_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
    )
    model.load_weights()
    return model


def get_trained_model(
    batch_size: int,
    device: str,
    model_config: DecoderConfig = _DEFAULT_DECODER_CONFIG,
) -> Decoder:
    """Load a pre-trained Decoder model from saved weights.

    Args:
        batch_size: Batch size for the model
        device: Device to load weights onto ("cpu", "cuda", or "mps")
        model_config: Decoder architecture configuration (must match saved weights)

    Returns:
        Decoder model with loaded weights
    """
    if device not in Devices:
        raise ValueError("device must be cpu, cuda, or mps")

    weights = torch.load(
        str(files("sentimentizer.data").joinpath("decoder_weights.pth")),
        map_location=torch.device(device=device),
    )
    # Infer dimensions from saved weights
    emb_shape = weights["embed_layer.weight"].shape
    d_model = weights["proj.weight"].shape[0]

    empty_embeddings = torch.zeros(emb_shape)
    model = Decoder(
        batch_size=batch_size,
        d_model=d_model,
        n_heads=model_config.n_heads,
        input_len=200,
        emb_weights=empty_embeddings,
        n_encoder_layers=model_config.n_encoder_layers,
        n_decoder_layers=model_config.n_decoder_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
    )

    model.load_state_dict(weights)

    return model
