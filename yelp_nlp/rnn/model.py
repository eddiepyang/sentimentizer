import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    """
    RNN model class
    """
    # weights are vocabsize x embedding length
    def __init__(self, emb_weights, batch_size, input_len):

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
        self.lstm = nn.LSTM(input_len, input_len)
        self.fc1 = nn.Linear(input_len, 1)
        self.fc2 = nn.Linear(emb_weights.shape[1], 1)

    def load_weights(self):
        self.embed_layer.load_state_dict({'weight': self.emb_weights})
        return self

    def forward(self, inputs, p=0.2, verbose=False):

        embeds = self.embed_layer(inputs)

        nn.Dropout2d(p=p, inplace=True)(embeds)

        if verbose:
            print('embedding shape %s' % (embeds.shape,))

        out, (hidden, cell) = self.lstm(embeds.permute(0, 2, 1))

        if verbose:
            print('lstm out shape %s' % (out.shape,))

        out = F.relu(self.fc1(out))
        if verbose:
            print('fc1 out shape %s' % (out.shape,))

        fout = self.fc2(out.view(1, -1, 100))

        if verbose:
            print('final %s' % (fout.shape,))

        return torch.sigmoid(
            torch.squeeze(fout)
        )
