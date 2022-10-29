import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self, maxlen, vocab_size, n_channels, hidden_size, output_size, implement_pe=False):
        super(CNN, self).__init__()
        self.maxlen = maxlen
        self.implement_pe = implement_pe
        self.embedding = nn.Linear(vocab_size, hidden_size)
        self.conv2 = torch.nn.Conv2d(1, n_channels, (2, hidden_size), padding=0)
        self.conv3 = torch.nn.Conv2d(1, n_channels, (3, hidden_size), padding=0)
        self.conv4 = torch.nn.Conv2d(1, n_channels, (4, hidden_size), padding=0)
        self.l2 = nn.Linear((maxlen-1)*n_channels, hidden_size//4)
        self.l3 = nn.Linear((maxlen-2)*n_channels, hidden_size//4)
        self.l4 = nn.Linear((maxlen-3)*n_channels, hidden_size//4)
        self.h2o = nn.Linear(hidden_size//4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pe = PositionalEncoding(hidden_size)

    def forward(self, input):
        '''
        input shape (bs, word_len, vocab_size)
        make it (bs, 1[channel], maxlen, vocab_size) first
        '''
        input = self.embedding(input) # (bs, word_len, hidden_size)
        if self.implement_pe:
            input = torch.transpose(input, 1, 0,)
            input = self.pe(input)  # (word_len, bs, hidden_size)
            input = torch.transpose(input, 1, 0,)
        input = self.padding(input)
        input = input.unsqueeze(1)
        h2 = F.relu(self.l2(F.relu(self.conv2(input)).reshape(1, -1)))
        h3 = F.relu(self.l3(F.relu(self.conv3(input)).reshape(1, -1)))
        h4 = F.relu(self.l4(F.relu(self.conv4(input)).reshape(1, -1)))
        output = F.relu(self.h2o(h2 + h3 + h4))
        output = self.softmax(output)
        return output

    def padding(self, x):
        # x shape: (1, word_len, hidden_size)
        _, word_len, hidden_size = x.shape
        return torch.cat((x, torch.zeros(size=(1, self.maxlen-word_len, hidden_size))), dim=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


if __name__ == '__main__':
    vocab_size = 3
    word_len = 4
    cnn = CNN(maxlen=5, vocab_size=vocab_size, n_channels=2, hidden_size=4, output_size=6, implement_pe=True)
    input = torch.rand(size=(1, word_len, vocab_size))
    a = cnn(input)
    print(a)

