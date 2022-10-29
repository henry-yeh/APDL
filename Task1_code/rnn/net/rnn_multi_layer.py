import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        assert n_layers > 1, 'Please implement RNN from rnn.py if the number of layers is 1.'
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(2*hidden_size, output_size)

        self.hidden_layers = [nn.Linear(2*hidden_size, hidden_size) for _ in range(n_layers-1)]


        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hiddens):
        combined = torch.cat((input, hiddens[0]), 1)
        hidden = self.i2h(combined)
        output_hiddens = [hidden]
        for i in range(0, len(self.hidden_layers)):
            combined = torch.cat((hidden, hiddens[i+1]), 1)
            hidden = self.hidden_layers[i](combined)
            output_hiddens.append(hidden)
        output = self.h2o(combined)
        output = self.softmax(output)
        return output, output_hiddens

    def initHidden(self):
        return [torch.zeros(1, self.hidden_size) for i in range(self.n_layers)]

