from importlib.resources import files

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import corpora

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    VALID_DEVICES,
    EmbeddingsConfig,
    RNNConfig,
)
from sentimentizer.extractor import new_embedding_weights

logger = new_logger(DEFAULT_LOG_LEVEL)

# Module-level singleton for B008 compliance — used as default in function signatures
_DEFAULT_RNN_CONFIG = RNNConfig()


class RNN(nn.Module):
    """Bidirectional LSTM for sentiment classification.

    Uses pre-trained GloVe embeddings, a multi-layer bidirectional LSTM that
    processes tokens in sequence order (batch_first=True), and concatenates
    the final forward and backward hidden states for classification.

    Args:
        emb_weights: Pre-trained embedding weights of shape (vocab_size, emb_dim)
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        verbose: Whether to log debug shapes
        dropout: Dropout probability
    """

    def __init__(
        self,
        emb_weights: torch.Tensor,
        hidden_size: int = RNNConfig.hidden_size,
        num_layers: int = RNNConfig.num_layers,
        verbose: bool = False,
        dropout: float = RNNConfig.dropout,
    ) -> None:
        super().__init__()
        emb_dim = emb_weights.shape[1]

        # Embedding layer with pre-trained GloVe weights
        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_dim)
        self.fc0 = nn.Linear(emb_dim, emb_dim)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Bidirectional LSTM with batch_first=True
        # Processes tokens in sequence order — word 1, word 2, ..., word 200
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Classification head: concatenated final hidden states → logit
        # hidden_size * 2 because bidirectional (forward + backward)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.verbose = verbose

        # Load embedding weights immediately
        self.embed_layer.load_state_dict({"weight": emb_weights})  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            inputs: Token IDs of shape (batch, seq_len)

        Returns:
            Logits of shape (batch,)
        """
        embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
        embeds = self.dropout_layer(embeds)
        if self.verbose:
            logger.info(f"embedding shape {embeds.shape}")

        embeds = F.relu(self.fc0(embeds), inplace=True)  # (B, seq_len, emb_dim)

        # LSTM processes tokens in order — no permute needed
        out, (hidden, cell) = self.lstm(embeds)  # out: (B, seq_len, hidden*2)
        if self.verbose:
            logger.info(f"lstm out shape {out.shape}")

        # hidden shape: (num_layers * 2, B, hidden_size)
        # Take final layer's forward and backward hidden states
        hidden_fwd = hidden[-2]  # last forward layer: (B, hidden_size)
        hidden_bwd = hidden[-1]  # last backward layer: (B, hidden_size)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, hidden*2)

        if self.verbose:
            logger.info(f"hidden cat shape {hidden_cat.shape}")

        logits = self.classifier(hidden_cat)  # (B, 1)
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
    model_config: RNNConfig = _DEFAULT_RNN_CONFIG,
) -> RNN:
    """Create a new RNN model with pre-trained GloVe embeddings.

    Args:
        dict_path: Path to the gensim dictionary file
        embeddings_config: Configuration for GloVe embeddings
        input_len: Maximum sequence length
        model_config: RNN architecture configuration (defaults from RNNConfig)
    """
    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = RNN(
        emb_weights=emb_t,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )
    return model


def get_trained_model(
    device: str,
    model_config: RNNConfig = _DEFAULT_RNN_CONFIG,
) -> RNN:
    """Load a pre-trained RNN model from saved weights.

    Args:
        device: Device to load weights onto ("cpu", "cuda", or "mps")
        model_config: RNN architecture configuration (must match saved weights)

    Returns:
        RNN model with loaded weights
    """
    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    weights_path = str(files("sentimentizer.data").joinpath("weights.pth"))

    try:
        weights = torch.load(
            weights_path, map_location=torch.device(device=device), weights_only=True
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Weights file not found at {weights_path}. "
            "Please train the model first: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        ) from e

    # Check if weights are from the new architecture (has 'classifier' keys)
    if "classifier.0.weight" not in weights:
        raise RuntimeError(
            "Saved weights are from a previous model architecture and are incompatible. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        )

    emb_shape = weights["embed_layer.weight"].shape
    hidden_size = weights["classifier.0.weight"].shape[1] // 2

    empty_embeddings = torch.zeros(emb_shape)
    model = RNN(
        emb_weights=empty_embeddings,
        hidden_size=hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )

    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        raise RuntimeError(
            "Saved weights are incompatible with the current model architecture. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        ) from e

    return model
