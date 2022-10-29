import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for language modeling.")
# -- train
    parser.add_argument('--n_epochs', type=int, default=3, help="The number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
# -- hyperparameters
    parser.add_argument('--lr', type=float, default=5., help='Learning rate')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads of multi-head attention')
    parser.add_argument('--d_hid', type=int, default=200,
        help='Dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', 
        help="Activation function, can be 'relu' or  'gelu'")
    parser.add_argument('--optimizer', type=int, default=1, 
        help='ID of the chosen optimizer')
    parser.add_argument('--pop_size', type=int, default=1, 
        help='Population size of GA')
    parser.add_argument('--max_iter', type=int, default=1, 
        help='Number of iterations to run GA')
    
# -- run setting
    parser.add_argument('--search', action='store_true', help='Search the best hyperparameter')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    opts = parser.parse_args(args)

    return opts
