import copy
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from text.positional_encoding import SinPosEncoding
from tokenizers import Tokenizer

# TODO: Dataset, batching and training

# TODO: pipeline for data processing

# TODO: remove asserts and raise exceptions

# TODO: shift decoder's inputs (target sequence) to the right by preprending
# BOS token. Can implement a Shift module to be used as part of a data pipeline.
# Example - Dataset entry : <s> A B C D E </s>
#           Target        :  A  B C D E </s>
#           Decoder input : <s> A B C D E

# TODO: torch.compile before training to improve performance

# NOTE: padding can be done on the left or right of each sequence. LLMs (which
# are decoder-only architectures) are typically padded on the left during
# inference or fine-tuning, since they are not pre-trained to continue
# sentences from <pad> tokens. Padding on the right might be ok for encoder or
# for sequence-to-sequence architectures, since training is typically carried
# out with padding (decoder-only architectures do not need padding since they
# can just truncate sequences up to a prefixed length from the training
# corpus). Padding on the right might also be safer for absolute positional
# encodings approaches.

# TODO: scaled_dot_product_attention utility function

# TODO: make TransformerDecoder and TransformerDecoderLayer agnostic of the use
# of an encoder: that is it should be possible to use TransformerDecoder to
# build a decoder-only architecture that does not use self-attention with the
# encoder's output. Possibly, we could just use TransformerEncoder as a mean of
# constructing decoder-olny architectures, by passing an `is_causal` argument
# to it.

# TODO: possibly integrate loss calculation within task heads (such as
# LanguageModelingHead)

def is_initializable(module: nn.Module) -> bool:
    return isinstance(module, tuple([nn.Linear, nn.LayerNorm]))

class Replicated(nn.Module):
    """ Wrapper module that stacks a given Module a number of times.
        The constructor tries to reinitialize the parameters of each copied
        layer by calling `reset_parameters()` for each child module
        recursively.
    """
    def __init__(
            self,
            layer: nn.Module,
            n_layers: int
            ) -> None:
        super().__init__()
        layers = [copy.deepcopy(layer) for i in range(n_layers)]
        self.stacked = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.apply(lambda m: m.reset_parameters() if is_initializable(m) else None)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for layer in self.stacked:
            x = layer(x, *args, **kwargs)
        return x

class MultiHeadAttention(nn.Module):
    """ Multi head attention module.
        Query, key and value vectors share the same dimension `dim`.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            inner_dim: int = None
            ) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        
        if not self.inner_dim:
            self.inner_dim = dim // n_heads

        # TODO: elaborate on not using biases (LayerNormalization adds biases
        # or cancels them out by subtracting mean?)
        self.qkv_proj = nn.Linear(dim, self.inner_dim * self.n_heads * 3, bias = False)
        self.out_proj = nn.Linear(self.inner_dim * n_heads, dim, bias = False)

    def _qkv_proj(self, query, key, value):
        if query is key and key is value:
            # Compute projection for query, key, value for all heads in parallel and split it.
            qkv = self.qkv_proj(query).chunk(3, dim=-1) # tuple of 3x(B, N, DIM)
        else:
            # weight.T \in R^{dim \times 3 * inner_dim * n_heads}
            d = self.inner_dim * self.n_heads
            q_proj_weight = self.qkv_proj.weight[0:d, :]
            k_proj_weight = self.qkv_proj.weight[d:2*d, :]
            v_proj_weight = self.qkv_proj.weight[2*d:, :]

            # No biases
            q = F.linear(query, q_proj_weight)
            k = F.linear(key, k_proj_weight)
            v = F.linear(value, v_proj_weight)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads, d = self.inner_dim), qkv)
        return q, k, v

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            causal: bool = False,
            key_mask: Tensor = None,
            query_mask: Tensor = None,
            ) -> Tensor:

        if (key_mask is None) != (query_mask is None):
            raise ValueError("Either both key_mask and query_mask must be None, or both must be provided.")

        q, k, v = self._qkv_proj(query, key, value)

        # Batched Query-Value matrix multiplications over the last two dims:
        # the remaining are considered as batch dimensions
        attn = torch.matmul(q, k.transpose(-1, -2))

        # Normalization: we scale by the sqrt of the dimension of each head because
        # QK^T computes, for each head, dot products with vectors of dimension
        # self.inner_dim. If the vectors were (independent and) randomly
        # distributed with mean 0 and unit variance then the variance of the
        # dot product would be self.inner_dim. So scaling by the standard
        # deviation is a sound normalization scheme.
        attn = attn / math.sqrt(self.inner_dim)


        # Masking:
        # We do not assume, in general, that masked positions appear on the
        # sides. There might be reasons to mask tokens within the boundaries of
        # a sequence: for example, sparse attention, etc.
        if key_mask is not None:

            assert key_mask.shape[1] == key.shape[1] # must match sequence length
            assert query_mask.shape[1] == query.shape[1] # must match sequence length

            # attention_mask = torch.full((n, n), float('-inf'), device=query.device)
            assert key_mask is not None and query_mask is not None

            if key_mask.dtype is not torch.bool:
                key_mask = key_mask.bool()

            if query_mask.dtype is not torch.bool:
                query_mask = query_mask.bool()

            # TODO: assert shape before unsqueezing masks
            # Add a new dimension at position 1 -> (B, 1, N)
            key_mask = key_mask.unsqueeze(1)
            query_mask = query_mask.unsqueeze(1)

            # The transpose produces the shape -> (B, N, 1)
            # The & operator is broadcasted along dimension 1 for the first
            # operand and along dimension 2 for the second. This replicates the
            # binary mask along the rows for the first operand and along the
            # columns for the second one, which virtually creates two batches
            # of matrices of size (B, N, N) where the second one is the
            # transpose of the first one. By 'logically-and' them together we
            # obtain the correct mask for each sequence in the batch
            mask = key_mask & query_mask.transpose(1, 2)
            mask = torch.where(~mask, float('-inf'), 0.0)

            # Add new 'heads' dimension for broadcasting -> (B, 1, N, N)
            # the attention matrix is (B, H, N, N) so the mask is broadcasted H
            # times along that dimension
            mask = mask.unsqueeze(1)
            attn = attn + mask

        if causal:
            # By masking the elements of the preactivation attention matrix to
            # -inf, the softmax automatically drops them to zero while
            # preserving the sum-to-one constraint. We can use a single
            # attention mask for this since it's shared among every sequence
            # (because of padding they all have the same length)
            n = query.shape[1]
            causal_mask = torch.full((n, n), float('-inf'), device=query.device).triu(diagonal=1)
            attn = attn + causal_mask

        # Row softmax
        attn = torch.softmax(attn, dim = -1)
        # Hack to prevent softmax from producing `nan`s when entire rows of the
        # activation matrix are "masked" with -inf. This should be better
        # approached with MaskedTensors, but they are still a propotype
        # feature. An alternative approach would be to use
        # torch.finf(attn.dtype).min as filling value instead of -inf, but this
        # would produce a uniform distribution instead of all zeros. These
        # values are not considered during computation due to column masking,
        # but they might interfere during the last projections.
        attn = torch.nan_to_num(attn, 0.0)

        # Value multiplication
        attn = torch.matmul(attn, v)

        # Concatenate heads
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        # Project back to dim (unnecessary if self.inner_dim * n_heads == dim)
        if self.inner_dim * self.n_heads != self.dim:
            attn = self.out_proj(attn)

        assert attn.shape[-1] == self.dim

        return attn

class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder or self.decoder, "Either an encoder or a decoder should be defined."

        self.dim = self.encoder.dim if self.encoder else self.decoder.dim

    def forward(self, enc_in: Tensor, dec_in: Tensor, enc_mask: Tensor = None, dec_mask: Tensor = None):
        if self.encoder:
            enc_in = self.encoder(enc_in, mask = enc_mask)
        if self.decoder:
            dec_in = self.decoder(dec_in, enc_in, dec_mask = dec_mask, enc_mask = enc_mask)
        return dec_in

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

class TransformerEncoder(Replicated):
    def __init__(self,*args, causal: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.causal = causal
        self.dim = self.stacked[0].dim

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        return super().forward(x, causal = self.causal, mask = mask)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, attention: nn.Module):
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.attention = attention
        if not self.attention:
            self.attention = MultiHeadAttention(dim, n_heads = 8, inner_dim = dim // 8)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim)
                )

    def forward(self, x, causal = False, mask = None):
        x = self.norm1(self.attention(x, x, x, causal = causal, query_mask = mask, key_mask = mask) + x) # Norm first = False (original paper)
        x = self.norm2(self.mlp(x) + x) # Norm first = False (original paper)
        return x

# TODO: need better naming for enc_out, dec_out enc_mask, dec_mask
# TODO: parameter order seems off, must think about it
class TransformerDecoder(Replicated):
    def forward(self, dec_out: Tensor, enc_out: Tensor, dec_mask: Tensor = None, enc_mask: Tensor = None) -> Tensor:
        return super().forward(dec_out, enc_out, dec_mask, enc_mask)

class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_dim: int,
            attention: nn.Module,
            ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim

        if not attention:
            attention = MultiHeadAttention(dim, n_heads = 8, inner_dim = dim // 8)

        self.self_attention = attention
        self.cross_attention = copy.deepcopy(attention)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim)
                )

    def forward(self, dec_out: Tensor, enc_out: Tensor, dec_mask: Tensor = None, enc_mask: Tensor = None) -> Tensor:
        dec_out = self.norm1(self.self_attention(dec_out, dec_out, dec_out, causal=True, query_mask = dec_mask, key_mask = dec_mask) + dec_out)
        dec_out = self.norm2(self.cross_attention(dec_out, enc_out, enc_out, query_mask = dec_mask, key_mask = enc_mask) + dec_out)
        dec_out = self.norm3(self.mlp(dec_out) + dec_out)
        return dec_out


class NLPTransformer(Transformer):
    def __init__(
            self,
            tokenizer: Tokenizer,
            *args,
            pos_encoding: nn.Module = SinPosEncoding(),
            embedding: nn.Module = None,
            padding: bool = True,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.pos_encoding = pos_encoding
        self.embedding = embedding
        self.padding = padding

        if not self.embedding:
            self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.dim)

    def forward(self, enc_in, dec_in, device='cpu'):

        # Tokenization
        tokenized_enc = self.tokenizer(enc_in, padding = self.padding)
        tokenized_dec = self.tokenizer(dec_in, padding = self.padding)
        enc_in = torch.tensor(tokenized_enc['input_ids'], device=device)
        dec_in = torch.tensor(tokenized_dec['input_ids'], device=device)

        # Padding masks
        enc_mask = torch.tensor(tokenized_enc['attention_mask'], device=device)
        dec_mask = torch.tensor(tokenized_dec['attention_mask'], device=device)

        # Embedding
        enc_in = self.embedding(enc_in)
        dec_in = self.embedding(dec_in)

        # Positional encoding
        enc_in = self.pos_encoding(enc_in)
        dec_in = self.pos_encoding(dec_in)

        return super().forward(enc_in, dec_in, enc_mask = enc_mask, dec_mask = dec_mask)

class PointwiseClassificationHead(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            in_dim: int,
            classes: int
            ) -> None:
        super().__init__()
        self.model = model
        self.classes = classes
        self.linear = nn.Linear(in_dim, classes)

    def forward(self, *args, log: bool = False, **kwargs) -> Tensor:
        out = self.model(*args, **kwargs) # (B, N, D)
        out = self.linear(out) # (B, N, C)

        # Broadcasted along (B, N) dimensions
        if log:
            out = torch.log_softmax(out, dim=-1)
        else:
            out = torch.softmax(out, dim=-1)
        return out

class LanguageModelingHead(PointwiseClassificationHead):
    def __init__(
            self,
            model: NLPTransformer
            ) -> None:
        super().__init__(model, model.dim, model.tokenizer.vocab_size)

