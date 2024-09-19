import torch
from torch import nn, Tensor


def extract_probs(dist: Tensor, target: Tensor) -> Tensor:
    """ Extract the probability of the target tokens from the input tensor 
        of distributions over the set of tokens.
    """
    # dist:     (B, N, Z) 
    # target:   (B, N)
    # result:   (B, N, Z) -> (B, N)
    return torch.gather(dist, -1, target.unsqueeze(-1))
