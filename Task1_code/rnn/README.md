#  Classifying Names with Handcrafted Neural Models

叶皓然

2027407057





## Introduction
This is the PyTorch code for the first project (PART II) of *Application Practice of Deep Learning*.


I handcraft neural models (i.e., Base RNN, Bidirectional RNN, Multi-layer RNN, GRU, CNN) with the most basic nn modules provided by Torch (i.e., nn.Linear, nn.Conv2d). I specially hold out a validation dataset to compare the performance of my models during training.


## For name classification


To train Base RNN:
```bash
python unitrain.py --model rnn
```

To train an RNN with a different hidden dimension (e.g. 256):

```bash
python unitrain.py --model rnn --n_hidden 256
```

To train an RNN with multiple hidden layers (e.g. 4):
```bash
python mltrain.py --n_layers 4
```

To train a Bidirectional Base RNN:
```bash
python bitrain.py
```

To train a Bidirectional Base RNN:
```bash
python bitrain.py
```

To train the specially designed CNN:
```bash
python cnntrain.py
```

To train the specially designed CNN with positional encoding:
```bash
python cnntrain.py --pe
```

To train Base GRU:
```bash
python unitrain.py --model gru --n_hidden 128
```

## For name generation


### See *char_rnn_generation.ipynb*, where RNN and GRU are implemented.


## Dependencies

* Python>=3.7, <=3.10
* NumPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm




