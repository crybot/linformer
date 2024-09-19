import torch
from torch import nn, Tensor

def perplexity(target_log_probs: Tensor) -> Tensor:
    # (B, N, Z) where Z is the vocabulary size
    # -> sum over sequence dimension (N)
    return torch.exp(-torch.mean(target_log_probs))

class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_log_probs):
        return perplexity(target_log_probs)
