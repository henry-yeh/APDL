import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        self.wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.w = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        r = torch.sigmoid(self.wr(combined))
        z = torch.sigmoid(self.wz(combined))
        hidden_reset = r * hidden
        combined2 = torch.cat((input, hidden_reset), 1)
        hidden_prime = torch.tanh(self.w(combined2))
        hidden = (1-z)*hidden + z * hidden_prime
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

