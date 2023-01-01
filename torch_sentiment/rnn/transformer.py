import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from gensim import corpora
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch_sentiment.extractor import new_embedding_weights
from torch_sentiment.logging_utils import new_logger
from torch_sentiment.rnn.config import (
    EmbeddingsConfig,
    TokenizerConfig,
    DEFAULT_LOG_LEVEL,
)

from importlib.resources import files


logger = new_logger(DEFAULT_LOG_LEVEL)


class TransformerSentiment(nn.Module):
    """model class"""

    def __init__(
        self,
        batch_size: int,
        input_len: int,
        emb_weights: torch.Tensor,  # weights are vocabsize x embedding length
        verbose: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        # vocab size in, hidden size out
        self.batch_size = batch_size
        self.emb_weights = emb_weights

        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1])
        self.fc0 = nn.Linear(emb_weights.shape[1], emb_weights.shape[1])

        self.dropout = dropout
        self.dropout_layer = nn.Dropout1d(p=self.dropout, inplace=True)
        # input of shape (seq_len, batch, input_size)
        # https://pytorch.org/docs/stable/nn.html
        self.transformer = TransformerModel(emb_weights.shape[0], input_len, 6, 6, 2)
        self.fc1 = nn.Linear(input_len, 1)
        self.fc2 = nn.Linear(emb_weights.shape[1], 1)
        self.verbose = verbose

    def load_weights(self):
        self.embed_layer.load_state_dict({"weight": self.emb_weights})  # type: ignore
        return self

    def forward(self, inputs: torch.Tensor):
        embeds = self.embed_layer(inputs)
        self.dropout_layer(embeds)
        if self.verbose:
            logger.info("embedding shape %s" % (embeds.shape,))
        embeds = F.relu(self.fc0(embeds))
        out = self.transformer(embeds.permute(0, 2, 1))
        if self.verbose:
            logger.info("lstm out shape %s" % (out.shape,))
        out = self.fc1(out)
        if self.verbose:
            logger.info("fc1 out shape %s" % (out.shape,))
        fout = self.fc2(out.permute(0, 2, 1))
        if self.verbose:
            logger.info("final %s" % (fout.shape,))

        return torch.squeeze(fout)

    def predict(self, converted_text: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            output = torch.from_numpy(converted_text)
            return torch.sigmoid(self.forward(output))


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,  # dimensions of embedding length
        nhead: int, # output dimensions
        d_hid: int, 
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def new_model(
    dict_path: str, embeddings_config: EmbeddingsConfig, batch_size: int, input_len: int
):
    # dict_yelp = corpora.Dictionary.load(dict_path)
    # embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    # emb_t = torch.from_numpy(embedding_matrix)
    # model = TransformerSentiment(batch_size, input_len, emb_weights)(batch_size=batch_size, input_len=input_len, emb_weights=emb_t)
    # model.load_weights()
    # return model
    return


def get_trained_model(batch_size: int, device: str) -> TransformerSentiment:
    """loads pre-trained model"""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")

    weights = torch.load(
        files("torch_sentiment.data").joinpath("weights.pth"),
        map_location=torch.device(device=device),
    )
    empty_embeddings = torch.zeros(weights["embed_layer.weight"].shape)
    model = RNN(
        batch_size=batch_size,
        input_len=TokenizerConfig.max_len,
        emb_weights=empty_embeddings,
    )
    model.load_state_dict(weights)

    return model
