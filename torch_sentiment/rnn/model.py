import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim import corpora
from typing import Optional

from torch_sentiment.rnn.extractor import new_embedding_weights
from torch_sentiment.logging_utils import new_logger
from torch_sentiment.rnn.config import EmbeddingsConfig, TokenizerConfig, LogLevels

from importlib.resources import files


logger = new_logger(LogLevels.debug.value)


class RNN(nn.Module):
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

        self.lstm = nn.LSTM(input_len, input_len)
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
        out, (hidden, cell) = self.lstm(embeds.permute(0, 2, 1))
        if self.verbose:
            logger.info("lstm out shape %s" % (out.shape,))
        out = self.fc1(out)
        if self.verbose:
            logger.info("fc1 out shape %s" % (out.shape,))
        fout = self.fc2(out.permute(0, 2, 1))
        if self.verbose:
            logger.info("final %s" % (fout.shape,))

        return torch.squeeze(fout)


def new_model(
    dict_path: str, embeddings_config: EmbeddingsConfig, batch_size: int, input_len: int
):
    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = RNN(batch_size=batch_size, input_len=input_len, emb_weights=emb_t)
    model.load_weights()
    return model


def get_trained_model(batch_size: int, input_len: int) -> RNN:

    """loads pre-trained model"""
    empty_embeddings = torch.zeros(
        4735, EmbeddingsConfig.emb_length
    )
    model = RNN(
        batch_size=batch_size, input_len=input_len, emb_weights=empty_embeddings
    )
    model.load_state_dict(
        torch.load(files("torch_sentiment.data").joinpath("weights.pth"))
    )
    return model
