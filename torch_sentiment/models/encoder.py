from importlib.resources import files

import numpy as np
import torch
import torch.nn.functional as F
from gensim import corpora
from torch import nn

from torch_sentiment import new_logger
from torch_sentiment.config import DEFAULT_LOG_LEVEL, EmbeddingsConfig, Devices
from torch_sentiment.extractor import new_embedding_weights

logger = new_logger(DEFAULT_LOG_LEVEL)


class Encoder(nn.Module):
    """model class"""

    def __init__(
        self,
        batch_size: int,
        input_len: int,
        d_model: int,
        n_heads: int,
        emb_weights: torch.Tensor,  # weights are vocabsize x embedding length
        verbose: bool = True,
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

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=3, norm=layer_norm
        )

        self.fc1 = nn.Linear(input_len, 1)
        self.fc2 = nn.Linear(emb_weights.shape[1], 1)
        self.verbose = verbose

    def load_weights(self):
        self.embed_layer.load_state_dict({"weight": self.emb_weights})  # type: ignore
        return self

    def forward(self, inputs: torch.Tensor):
        embeds = self.embed_layer(inputs)
        self.dropout_layer(embeds)

        logger.debug("embedding shape %s" % (embeds.shape,))
        embeds = F.relu(self.fc0(embeds))
        encoded_out = self.encoder(embeds.permute(0, 2, 1))

        logger.debug("lstm out shape %s" % (encoded_out.shape,))
        out = self.fc1(encoded_out)
        logger.debug("fc1 out shape %s" % (out.shape,))
        fout = self.fc2(out.permute(0, 2, 1))
        logger.debug("final %s" % (fout.shape,))

        return torch.squeeze(fout)

    def predict(self, converted_text: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            output = torch.from_numpy(converted_text)
            return torch.sigmoid(self.forward(output))


def new_model(
    dict_path: str, embeddings_config: EmbeddingsConfig, batch_size: int, input_len: int
):
    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = Encoder(
        batch_size=batch_size,
        d_model=200,
        n_heads=4,
        input_len=input_len,
        emb_weights=emb_t,
    )
    model.load_weights()
    return model


def get_trained_model(batch_size: int, device: str) -> Encoder:
    """loads pre-trained model"""
    if device not in Devices:
        raise ValueError("device must be cpu, cuda, or mps")

    weights = torch.load(
        str(files("torch_sentiment.data").joinpath("embed_weights.pth")),
        map_location=torch.device(device=device),
    )
    empty_embeddings = torch.zeros(weights["embed_layer.weight"].shape)
    model = Encoder(
        batch_size=batch_size,
        d_model=200,
        n_heads=4,
        input_len=200,
        emb_weights=empty_embeddings,
    )

    model.load_state_dict(weights)

    return model
