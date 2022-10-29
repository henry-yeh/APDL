import torch
from torch import nn
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import parse_data
from utils import randomTrainingExample, categoryFromOutput, timeSince
from net.birnn import RNN
from evaluate import evaluate

category_lines, all_categories, all_letters, test_lines, train_lines = parse_data()
n_letters = len(all_letters)
n_hidden = 128
n_categories = len(all_categories)
rnn = RNN(n_letters, n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.03 # If you set this too high, it might explode. If too low, it might not learn


layer = nn.LogSoftmax(dim=1)
bs=16

def train(category_tensor, line_tensor, iter):

    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output1, hidden = rnn(line_tensor[i], hidden)

    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]-1, -1, -1):
        output2, hidden = rnn(line_tensor[i], hidden)

    output = layer(output1 + output2)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    if iter % bs == 0:
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate/bs)
        rnn.zero_grad()

    return output, loss.item()


import time
import math

n_iters = 50000
print_every = 5000
plot_every = 500



# Keep track of losses for plotting
current_loss = 0
all_losses = []


accuracy_train = []
accuracy_val = []
start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
    output, loss = train(category_tensor, line_tensor, iter)
    current_loss += loss

    # Print iter number, loss, name and guess
    # if iter % print_every == 0:
    #     guess, guess_i = categoryFromOutput(output, all_categories)
    #     correct = '✓' if guess == category else '✗ (%s)' % category
    #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        print('evaluating...')
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        rnn.eval()
        print('On val set:')
        val_ac = evaluate(test_lines, rnn, all_categories)
        print('On training set:')
        train_ac = evaluate(train_lines, rnn, all_categories)
        print()
        rnn.train()
        accuracy_train.append(train_ac)
        accuracy_val.append(val_ac)


print(all_losses)
print()
print(accuracy_train)
print(accuracy_val)
print(max(accuracy_val))

plt.figure()
plt.plot(all_losses)
plt.show()