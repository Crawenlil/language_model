from src.utils import (
    UNK_STR, UNK
)

from src.utils import PAD
import torch


def get_language_model(conf, w2i, i2w):
    models = {
        'LSTMLanguageModel': LSTMLanguageModel,
    }
    return models[conf['model']](conf, w2i, i2w)


class LanguageModel(torch.nn.Module):
    def __init__(self, conf, word2index, index2word):
        super(LanguageModel, self).__init__()
        self.name = 'LSTMLanguageModel_{}.pt'.format(conf['embedding_dim'])
        self.w2i = word2index
        self.i2w = index2word

    def word2index(self, word):
        return self.w2i[word] if word in self.w2i else UNK

    def index2word(self, idx):
        return self.i2w[idx] if idx in self.i2w else UNK_STR


class LSTMLanguageModel(LanguageModel):
    def __init__(self, conf, word2index, index2word):
        super(LSTMLanguageModel, self).__init__(conf, word2index, index2word)

        self.embedding = torch.nn.Embedding(
            num_embeddings=conf['vocab_size'],
            embedding_dim=conf['embedding_dim'],
            padding_idx=PAD
        )
        self.lstm = torch.nn.LSTM(
            input_size=conf['embedding_dim'],
            hidden_size=conf['hidden_size'],
            num_layers=conf['lstm_num_layers'],
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        self.fc = torch.nn.Linear(conf['hidden_size'], conf['vocab_size'])

    def forward(self, X):
        X_lengths = (X != 0).sum(axis=-1)
        X = self.embedding(X)
        # pack sequence -> don't process pads during lstm step
        X = torch.nn.utils.rnn.pack_padded_sequence(
            X,
            X_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        X, _ = self.lstm(X)
        # unpack
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # Select outputs from last time step
        X = X[torch.arange(len(X_lengths)), X_lengths-1]
        X = self.fc(X)
        return X


class TransformerLanguageModel(LanguageModel):
    def __init__(self, conf):
        super(TransformerLanguageModel, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

    def forward(self, X):
        return self.transformer(X)
