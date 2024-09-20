import torch
from torch import Tensor, nn

def sinusoidal_pos_encoding(x: Tensor) -> Tensor:
    """ Returns the positional embedding computed as a combination of sine and
    cosine functions of the input dimensions.

    arguments:
    x   --  a batched tensor of sequences with shape (B, N, D)
    """
    # TODO: maybe it's okay to have > 3 due to broadcasting
    assert x.dim() == 3, "Expecting a 3D tensor of shape (B, N, D)"
    dim = x.shape[-1]
    n = x.shape[-2]

    pos = torch.arange(n, device=x.device).unsqueeze(1)
    i = torch.arange(dim, device=x.device)
    i_even = i % 2 == 0
    i_odd = ~i_even

    pe_sin = torch.sin(pos / 10000 ** (i[i_even]/ dim))
    pe_cos = torch.cos(pos / 10000 ** ((i[i_odd] - 1) / dim))
    
    pe = torch.cat([pe_sin, pe_cos], dim=-1)
    return pe.view(x.shape[1:])

class SinPosEncoding(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + sinusoidal_pos_encoding(x)

