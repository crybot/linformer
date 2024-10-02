import copy
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from text.positional_encoding import SinPosEncoding
from tokenizers import Tokenizer
from typing import Optional, Union

# TODO: pipeline for data processing

# TODO: remove asserts and raise exceptions

# TODO: possibly integrate loss calculation within task heads (such as
# LanguageModelingHead)

# TODO: define and annotate class parameters

# TODO: MaskedTensor interface

def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        dim = None
        ) -> Tensor:
    if not dim:
        dim = k.shape[-1]

    # Batched Query-Value matrix multiplications over the last two dims:
    # the remaining are considered as batch dimensions
    attn = torch.matmul(q, k.transpose(-1, -2))

    # Normalization: we scale by the sqrt of the dimension of each head because
    # QK^T computes, for each head, dot products with vectors of dimension
    # inner_dim. If the vectors were (independent and) randomly
    # distributed with mean 0 and unit variance then the variance of the
    # dot product would be inner_dim. So scaling by the standard
    # deviation is a sound normalization scheme.
    attn = attn / math.sqrt(dim)

    if attention_mask is not None:
        attn = attn + attention_mask

    # Row softmax
    attn = torch.softmax(attn, dim = -1)

    # Hack to prevent softmax from producing `nan`s when entire rows of the
    # activation matrix are "masked" with -inf. This should be better
    # approached with MaskedTensors, but they are still a propotype
    # feature. An alternative approach would be to use
    # torch.finfo(attn.dtype).min as filling value instead of -inf, but this
    # would produce a uniform distribution instead of all zeros. These
    # values are not considered during computation due to column masking,
    # but they might interfere during the last projections.

    # NOTE: disabled because it breaks gradient flow
    # attn = torch.nan_to_num(attn, 0.0)

    # Value multiplication
    attn = torch.matmul(attn, v)

    return attn

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

        # fill_value = float('-inf')
        fill_value = torch.finfo(q.dtype).min
        mask = None
        if key_mask is not None:

            assert key_mask.shape[1] == key.shape[1] # must match sequence length
            assert query_mask.shape[1] == query.shape[1] # must match sequence length

            assert key_mask is not None and query_mask is not None

            if key_mask.dtype is not torch.bool:
                key_mask = key_mask.bool()

            if query_mask.dtype is not torch.bool:
                query_mask = query_mask.bool()

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
            mask = torch.where(~mask, fill_value, 0.0)

            # Add new 'heads' dimension for broadcasting -> (B, 1, N, N)
            # the attention matrix is (B, H, N, N) so the mask is broadcasted H
            # times along that dimension
            mask = mask.unsqueeze(1)

        if causal:
            # By masking the elements of the preactivation attention matrix to
            # -inf, the softmax automatically drops them to zero while
            # preserving the sum-to-one constraint. We can use a single
            # attention mask for this since it's shared among every sequence
            # (because of padding they all have the same length)
            n = query.shape[1]
            causal_mask = torch.full((n, n), fill_value, device=query.device).triu(diagonal=1)
            if mask is not None:
                mask = mask + causal_mask
            else:
                mask = causal_mask

        attn = scaled_dot_product_attention(q, k, v, attention_mask=mask, dim=self.inner_dim)

        # Concatenate heads
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        # Project back to dim (unnecessary if self.inner_dim * n_heads == dim)
        if self.inner_dim * self.n_heads != self.dim:
            attn = self.out_proj(attn)

        assert attn.shape[-1] == self.dim
        return attn

class LinformerAttention(MultiHeadAttention):
    """ Multi head attention with linear projections on K and V
    """
    # TODO: rename k and sequence_length
    def __init__(self, *args, k: int, sequence_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_dim = k
        self.max_length = sequence_length

        # Using Linear so that it automatically handles initialization
        self.E = nn.Linear(self.max_length, self.proj_dim, bias=False)
        self.F = nn.Linear(self.max_length, self.proj_dim, bias=False)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            causal: bool = False, # here just for compatibility reasons
            key_mask: Tensor = None,
            query_mask: Tensor = None,
            full = False,
            ) -> Tensor:
        if (key_mask is None) != (query_mask is None):
            raise ValueError('Either both key_mask and query_mask must be None, or both must be provided.')

        if causal:
            raise ValueError('Warning: causal masking is not supported by the Linformer attention')

        q, k, v = self._qkv_proj(query, key, value)

        # TODO: mask before projecting on (query, key, value)
        if query_mask is not None:
            q = q.masked_fill(~query_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)

        # Share same mask for K and V
        if key_mask is not None:
            k = k.masked_fill(~key_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)
            v = v.masked_fill(~key_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)

        # Broadcast E @ K and F @ V over batch and head dimensions
        if not full:
            proj_k = self.E.weight
            proj_v = self.F.weight

            if key.shape[1] < self.max_length:
                proj_k = proj_k[:, :key.shape[1]]
                proj_v = proj_v[:, :key.shape[1]]

            k = torch.matmul(proj_k, k)
            v = torch.matmul(proj_v, v)

        attn = scaled_dot_product_attention(q, k, v, dim = self.inner_dim)

        # Concatenate heads
        attn = rearrange(attn, 'b h n d -> b n (h d)')

        # Project back to dim (unnecessary if self.inner_dim * n_heads == dim)
        if self.inner_dim * self.n_heads != self.dim:
            attn = self.out_proj(attn)

        assert attn.shape[-1] == self.dim
        return attn

class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.dim = self.encoder.dim if self.encoder else self.decoder.dim

    def forward(
            self,
            enc_in: Tensor,
            dec_in: Tensor,
            enc_mask: Tensor = None,
            dec_mask: Tensor = None,
            return_enc_output: bool = False,
            ):
        enc_out = self.encoder(enc_in, mask = enc_mask)
        dec_out = self.decoder(dec_in, enc_out, dec_mask = dec_mask, enc_mask = enc_mask)

        if return_enc_output:
            return enc_out, dec_out

        return dec_out

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
                nn.Linear(dim, mlp_dim, bias = False),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim, bias = False)
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
            cross_attention: nn.Module = None,
            ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim

        if not attention:
            attention = MultiHeadAttention(dim, n_heads = 8, inner_dim = dim // 8)

        self.self_attention = attention

        # The user can provide a different attention mechanism for the
        # cross-attention: this is useful in cases in which faster attention
        # mechanism cannot be applied to decoding self-attentions because they
        # are not compatible with causal masking (e.g. Linformer)
        if cross_attention:
            self.cross_attention = copy.deepcopy(cross_attention)
        else:
            self.cross_attention = copy.deepcopy(attention)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim, bias = False),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim, bias = False)
                )

    def forward(self, dec_out: Tensor, enc_out: Tensor, dec_mask: Tensor = None, enc_mask: Tensor = None) -> Tensor:
        dec_out = self.norm1(self.self_attention(dec_out, dec_out, dec_out, causal=True, query_mask = dec_mask, key_mask = dec_mask) + dec_out)
        dec_out = self.norm2(self.cross_attention(dec_out, enc_out, enc_out, query_mask = dec_mask, key_mask = enc_mask) + dec_out)
        dec_out = self.norm3(self.mlp(dec_out) + dec_out)
        return dec_out


class NLPTransformer(Transformer):
    def __init__(
            self,
            *args,
            pos_encoding: nn.Module = SinPosEncoding(),
            embedding: nn.Module = None,
            tokenizer: Tokenizer = None,
            vocab_size: int = None,
            padding: bool = True,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.pos_encoding = pos_encoding
        self.embedding = embedding
        self.padding = padding
        self.vocab_size = vocab_size

        # TODO: if tokenizer is None ??
        if not self.vocab_size:
            self.vocab_size = self.tokenizer.vocab_size

        if not self.embedding:
            self.embedding = nn.Embedding(self.vocab_size, self.dim)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None, device='cpu', **kwargs):
        # Tokenization
        if self.tokenizer:
            # TODO: return_tensors='pt': how to allocate them on device?
            tokenized_src = self.tokenizer(src, padding = self.padding)
            tokenized_tgt = self.tokenizer(tgt, padding = self.padding)
            enc_in = torch.tensor(tokenized_src['input_ids'], device=device)

            # Tokenization encloses the token ids with <s> ... </s>. To produce the
            # input to the decoder it's sufficient to drop the last token.
            # The target sequence is computed by dropping only the first token.
            dec_in = torch.tensor(tokenized_tgt['input_ids'], device=device)[..., :-1] # Drop last 
            dec_tgt = torch.tensor(tokenized_tgt['input_ids'], device=device)[..., 1:] # Drop first # TODO

            # Padding masks
            enc_mask = torch.tensor(tokenized_src['attention_mask'], device=device)
            dec_mask = torch.tensor(tokenized_tgt['attention_mask'], device=device)[..., :-1] # Drop last
            # TODO: compute tgt_mask for loss

        # Already tokenized: preparing inputs and masks
        else:
            enc_in = src
            dec_in = tgt

        # Embedding
        enc_in = self.embedding(enc_in)
        dec_in = self.embedding(dec_in)

        # Positional encoding
        enc_in = self.pos_encoding(enc_in)
        dec_in = self.pos_encoding(dec_in)

        pred = super().forward(enc_in, dec_in, enc_mask = src_mask, dec_mask = tgt_mask, **kwargs)
        return pred

    def decode(self, enc_out, dec_in, enc_mask = None, dec_mask = None):
        dec_in = self.embedding(dec_in) # TODO: can optimize
        dec_in = self.pos_encoding(dec_in)
        pred = self.decoder(dec_in, enc_out, dec_mask = dec_mask, enc_mask = enc_mask)
        return pred

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

    def lsm(self, out: Tensor, log: bool = False) -> Tensor:
        out = self.linear(out) # (B, N, C)

        # Broadcasted along (B, N) dimensions
        if log:
            out = torch.log_softmax(out, dim=-1)
        else:
            out = torch.softmax(out, dim=-1)
        return out

    def forward(self, *args, log: bool = False, **kwargs) -> Tensor:
        out = self.model(*args, **kwargs) # (B, N, D)
        return self.lsm(out, log = log)

class LanguageModelingHead(PointwiseClassificationHead):
    def __init__(
            self,
            model: NLPTransformer,
            loss_fn: nn.Module = None,
            ) -> None:
        super().__init__(model, model.dim, model.vocab_size)

    def generate(
            self,
            src: Tensor,
            tokenizer: Tokenizer,
            inputs: Optional[Tensor] = None,
            src_mask: Optional[Tensor] = None,
            max_length: int = 200,
            decode = True
            ) -> Union[Tensor, list[str]]:
        """ Implements a very rough version of greedy decoding """
        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        if inputs is None:
            inputs = torch.full((src.shape[0], 1), bos_token_id, dtype=torch.long, device=src.device)

        batch_size = inputs.shape[0]
        eos_flags = torch.zeros(batch_size)
        final_outputs = torch.full((batch_size, max_length), pad_token_id)

        self.model.eval()
        with torch.no_grad():
            # Wastes one decoding iteration but the code is much nicer
            tgt_mask = torch.ones_like(inputs)
            enc_out, _ = self.model.forward(src, inputs, src_mask = src_mask, tgt_mask = tgt_mask, return_enc_output=True)

            for t in range(max_length):
                dec_mask = torch.ones_like(inputs)
                dec_out = self.model.decode(enc_out, inputs, enc_mask = src_mask, dec_mask = dec_mask)
                dec_out = self.lsm(dec_out)[:, -1:, :] # Consider only last predicted token
                next_tokens = torch.argmax(dec_out, dim=-1)
                inputs = torch.cat([inputs, next_tokens], dim=-1)

                for i in range(batch_size):
                    if not eos_flags[i]:
                        final_outputs[i, t] = next_tokens[i]

                    # Update EOS flags where EOS is generated
                    if next_tokens[i] == eos_token_id:
                        eos_flags[i] = True

                # Stop decoding early if all sequences have hit EOS
                if eos_flags.all():
                    break

        if decode:
            return tokenizer.batch_decode(final_outputs, skip_special_tokens=True)
       
        return final_outputs

