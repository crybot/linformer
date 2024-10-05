import torch
from torch import Tensor
from typing import Union
import numpy as np

def pad_positions(tokens: Tensor, pad_token_id: int) -> tuple[Tensor]:
    """ 
    Return positions of pad_token_id over the input tokens tensor as a tuple of
    tensors, one for each input dimension.
    """
    return (tokens == pad_token_id).nonzero(as_tuple=True)

# TODO: allow contigous periods extraction (need to split tokens on period_token_id)
# TODO: resample extracts of size <= min_size (size=1 produces nans)
def random_tokens_extract(
        tokens: Tensor,
        max_length: int = 150
        ) -> Tensor:
    """
    Return a random contigous extract of the input tokens.
    """
    start_idx = np.random.choice(tokens.shape[-1], replace=False)

    # TODO: sample different location for each batch element

    return tokens[:, start_idx : start_idx + max_length]

def random_text_extract(
        text: str,
        max_periods: int = 3,
        max_length: int = 150,
        max_retries: int = 3
        ) -> list[str]:
    """
    Return a random contigous extract of the original input text.
    """
    periods = text.split('.')

    retries = 0

    while retries < max_retries:
        start_idx = np.random.choice(len(periods), replace=False)
        start = periods[start_idx]
        period_length = len(start.split())
        retries += 1

        if period_length <= max_length:
            break

    if period_length > max_length and retries == max_retries:
        raise Error(f"Could not find a period shorter than {max_length} words")

    total_length = period_length
    extract = [start]
    period_idx = start_idx + 1

    while len(extract) < max_periods and period_idx < len(periods):
        next_period = periods[period_idx]
        period_length = len(next_period.split())

        if total_length + period_length <= max_length:
            extract.append(next_period)
            total_length += period_length
            period_idx += 1
        else:
            break

    return ' '.join(extract)

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
        pad_token_id = None,
        return_mask = False
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
    mask_positions = torch.bernoulli(torch.ones_like(tokens) * mask_p).bool()
    # Do not mask first token of each sentence
    mask_positions[..., 0] *= not ignore_first
    # Do not mask last token of each sentence (if there is no pad)
    mask_positions[..., -1] *= not ignore_last

    # Retrieve pad positions over sentences (returns a tuple of tensors for each dim)
    if ignore_padding:
        assert pad_token_id is not None
        pad_index = pad_positions(tokens, pad_token_id)
        (*batch_idx, seq_idx) = pad_index
        # Do not mask <pad> tokens
        mask_positions[pad_index] = False
        # Do not mask last token before <pad> (i.e. </s>)
        mask_positions[(*batch_idx, seq_idx - 1)] = False

    # Apply mask
    masked[mask_positions] = mask_token_id

    if return_mask:
        return masked, mask_positions

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

def max_non_padded_length(sequence: Tensor, pad_token: int = 1) -> int:
    """ TODO """
    non_padded_mask = sequence != pad_token
    # Sum along the sequence dimension (N) to count non-padding tokens for each sequence
    lengths = non_padded_mask.sum(dim=1)

    # Find the maximum length in the batch
    return lengths.max().item()

def trim_batch_pad_tokens(inputs: tuple[Tensor], pad_token: int = 1) -> Tensor:
    """ TODO """
    src, tgt, src_mask, tgt_mask = inputs
    max_src_length = max_non_padded_length(src, pad_token = pad_token)
    max_tgt_length = max_non_padded_length(tgt, pad_token = pad_token)

    src, src_mask = src[:, :max_src_length], src_mask[:, :max_src_length]
    tgt, tgt_mask = tgt[:, :max_tgt_length], tgt_mask[:, :max_tgt_length]

    return src, tgt, src_mask, tgt_mask

def encoder_decoder_inputs(
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor
        ) -> tuple[Tensor]:
    """ Return the appropriate inputs for an encoder-decoder model:

        Expected arguments:
        - src:        source sequence enclosed in <s> </s> possibly padded
        - tgt:        target sequence enclosed in <s> </s> and possibly padded
        - src_mask:   source mask
        - tgt_mask:   target mask

        Returns tuple containing in order:
        - encoder input
        - decoder input
        - target sequence
        - encoder input mask
        - decoder input mask
        - target mask
    """
    return src, tgt[..., :-1], tgt[..., 1:], src_mask.bool(), tgt_mask[..., :-1].bool(), tgt_mask[..., 1:].bool()

