import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    """
    RNN model class
    """

    # weights are vocabsize x embedding length
    def __init__(
        self,
        emb_weights: torch.tensor,
        batch_size: int,
        input_len: int,
        verbose: bool = False
    ):

        super().__init__()
        # vocab size in, hidden size out
        self.batch_size = batch_size
        self.embed_layer = nn.Embedding(
            emb_weights.shape[0],
            emb_weights.shape[1]
        )
        self.emb_weights = emb_weights
        # input of shape (seq_len, batch, input_size)
        # https://pytorch.org/docs/stable/nn.html
        self.fc0 = nn.Linear(emb_weights.shape[1], emb_weights.shape[1])
        self.lstm = nn.LSTM(input_len, input_len)
        self.fc1 = nn.Linear(input_len, 1)
        self.fc2 = nn.Linear(emb_weights.shape[1], 1)
        self.verbose = verbose

    def load_weights(self):
        self.embed_layer.load_state_dict({'weight': self.emb_weights})
        return self

    def forward(
        self,
        inputs: torch.tensor,
        p: float = 0.2
    ):

        embeds = self.embed_layer(inputs)

        nn.Dropout2d(p=p, inplace=True)(embeds)

        if self.verbose:
            print('embedding shape %s' % (embeds.shape,))

        embeds = F.relu(
            self.fc0(embeds)
        )

        out, (hidden, cell) = self.lstm(embeds.permute(0, 2, 1))

        if self.verbose:
            print('lstm out shape %s' % (out.shape,))

        out = self.fc1(out)
        if self.verbose:
            print('fc1 out shape %s' % (out.shape,))

        fout = self.fc2(out.permute(0, 2, 1))

        if self.verbose:
            print('final %s' % (fout.shape,))

        return torch.squeeze(fout)
