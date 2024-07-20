import torch
from torch import Tensor
from typing import Union

def pad_positions(tokens: Tensor, pad_token_id: int) -> tuple[Tensor]:
    """ 
    Return positions of pad_token_id over the input tokens tensor as a tuple of
    tensors, one for each input dimension.
    """
    return (tokens == pad_token_id).nonzero(as_tuple=True)

def random_extract(text: str, max_periods: int = 3, max_length: int = 150) -> list[str]:
    """
    Return a random extract of the original input text.
    """
    pass

"""
Note: We apply random_mask to already tokenized sentences because otherwise
masking tokens would be non trivial: how do you split words, punctuation,
special characters, etc.
"""
def random_mask(
        tokens: Tensor,
        mask_token_id: int,
        mask_p: float = 0.1,
        ignore_first = True,
        ignore_last = True,
        ignore_padding = True,
        pad_token_id = None
        ) -> tuple[Tensor, Tensor]:
    """
    Randomly mask a subset of the input tokens.
    `tokens` must be a 1D or 2D Tensor.

    Return: a new set of tokens with randomly applied masks, and a tuple
    containing a 1-D Tensor for each dimension of the `tokens` Tensor
    containing the indices of the masked tokens (see
    torch.nonzero(as_tuple=False)).
    """

    assert tokens.dim() in [1, 2]

    masked = torch.clone(tokens)
    mask_positions = torch.bernoulli(torch.ones_like(tokens) * mask_p)
    # Do not mask first token of each sentence
    mask_positions[..., 0] *= int(not ignore_first)
    # Do not mask last token of each sentence (if there is no pad)
    mask_positions[..., -1] *= int(not ignore_last)

    # Retrieve pad positions over sentences (returns a tuple of tensors for each dim)
    pad_index = pad_positions(tokens, pad_token_id)
    (*batch_idx, seq_idx) = pad_index
    # Do not mask <pad> tokens
    mask_positions[pad_index] *= int(not ignore_padding)
    # Do not mask last token before <pad> (i.e. </s>)
    mask_positions[(*batch_idx, seq_idx - 1)] *= int(not ignore_padding)

    # Apply mask
    masked[mask_positions == 1.0] = mask_token_id

    return masked, mask_positions.nonzero(as_tuple=False)

def mask_fill(model, input_ids: Tensor, masked_index = None, mask_token_id : int = None, top_k: int = 5) -> list[Tensor]:
    """
    Return a (batched) list of the most probable predictions for a masked input.
    """
    predictions = [[] for _ in range(input_ids.shape[0])]
    logits = model(input_ids).logits # TODO: what if I don't take the logits? Are they already softmaxed?

    if masked_index is None:
        assert mask_token_id is not None
        masked_index = (input_ids == mask_token_id).nonzero(as_tuple=False)

    # Loop over the masked positions
    for (batch_idx, mask_pos) in masked_index:
        topk = logits[batch_idx, mask_pos].topk(5)
        predictions[batch_idx].append(topk.indices)

    return predictions
