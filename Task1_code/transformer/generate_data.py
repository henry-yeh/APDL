import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


if __name__ == '__main__':
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>']) 

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    print('Training data shape: ', train_data.shape)
    val_data = batchify(val_data, eval_batch_size)
    print('Validating data shape: ', val_data.shape)
    test_data = batchify(test_data, eval_batch_size)
    print('Test data shape: ', test_data.shape)

    torch.save(train_data, './data/train_data.pt')
    torch.save(val_data, './data/val_data.pt')
    torch.save(test_data, './data/test_data.pt')