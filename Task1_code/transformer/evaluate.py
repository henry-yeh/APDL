import torch
from torch import nn, Tensor
from net import generate_square_subsequent_mask
from utils import get_batch
import math
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
ntokens = 28782

def evaluate(model: nn.Module, eval_data: Tensor, bs) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    bptt = bs
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in tqdm(range(0, eval_data.size(0) - 1, bptt)):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

if __name__ == '__main__':
    model = torch.load('./pretrained/best_model.pt', map_location=device)
    eval_data = torch.load('./data/test_data.pt')
    val_loss = evaluate(model, eval_data, bs=128)
    try:
        val_ppl = math.exp(val_loss)
    except:
        val_ppl = float('inf')
    print('ppl on test set:', val_ppl)