import torch
from torch import nn, Tensor

def n_grams(tokens: Tensor, n: int) -> Tensor:
     # Get the sequence length
    seq_length = tokens.size(1)

    # Stack slices of the input tensor to create n-grams
    ngrams = torch.stack([tokens[:, i:seq_length - n + i + 1] for i in range(n)], dim=-1)

    return ngrams

def perplexity(target_log_probs: Tensor) -> Tensor:
    # (B, N, Z) where Z is the vocabulary size
    # -> sum over sequence dimension (N)
    return torch.exp(-torch.mean(target_log_probs))

def bleu_score(candidate: Tensor, target: Tensor, max_n: int = 4):
    raise NotImplementedError("Not implemented")
    for n in range(1, max_n + 1):
        cand_ngrams = n_grams(candidate, n)
        target_ngrams = n_grams(target, n)

class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_log_probs):
        return perplexity(target_log_probs)
