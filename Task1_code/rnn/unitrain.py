import torch
from torch import nn
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import time

from utils import parse_data
from utils import randomTrainingExample, categoryFromOutput, timeSince
from net.rnn import RNN
from net.gru import GRU
from evaluate import evaluate

def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--model', type=str, default='rnn')
    opts = parser.parse_args(args)
    return opts

opts = get_options()


category_lines, all_categories, all_letters, test_lines, train_lines = parse_data()
n_letters = len(all_letters)
n_hidden = opts.n_hidden
n_categories = len(all_categories)

model = {
    'gru': GRU,
    'rnn': RNN
}.get(opts.model)

rnn = model(n_letters, n_hidden, n_categories)


criterion = nn.NLLLoss()
learning_rate = 0.03 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor, iter):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    if iter % bs == 0:
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate/bs)
        rnn.zero_grad()

    return output, loss.item()



n_iters = 100000
print_every = 5000
plot_every = 500



# Keep track of losses for plotting
current_loss = 0
all_losses = []



accuracy_train = []
accuracy_val = []
start = time.time()

bs = 16

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