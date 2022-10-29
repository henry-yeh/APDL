import torch
from torch import nn
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

from utils import parse_data
from utils import randomTrainingExample, categoryFromOutput, timeSince
from net.cnn import CNN
from evaluate import cnn_evaluate

def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe', action='store_true')
    opts = parser.parse_args(args)
    return opts

opts = get_options()

category_lines, all_categories, all_letters, test_lines, train_lines = parse_data()
n_letters = len(all_letters)
n_hidden = 128
n_categories = len(all_categories)
print(n_categories)
cnn = CNN(
    maxlen=19, 
    vocab_size=57, 
    n_channels=4, 
    hidden_size=128, 
    output_size=len(all_categories), 
    implement_pe=opts.pe
    )

criterion = nn.NLLLoss()
learning_rate = 0.03 # If you set this too high, it might explode. If too low, it might not learn
bs=16
def train(category_tensor, line_tensor, iter):
    # input shape (bs, word_len, vocab_size)
    

    line_tensor = line_tensor.transpose(0, 1)
    output = cnn(line_tensor)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    if iter % bs == 0:
        for p in cnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate/bs)
        cnn.zero_grad()

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

for iter in tqdm(range(1, n_iters + 1)):
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
        cnn.eval()
        print('On val set:')
        val_ac = cnn_evaluate(test_lines, cnn, all_categories)
        print('On training set:')
        train_ac = cnn_evaluate(train_lines, cnn, all_categories)
        print()
        cnn.train()
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