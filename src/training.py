import torch
from torch import nn
from models.transformers import Transformer, MultiHeadAttention
from models.transformers import TransformerEncoder, TransformerEncoderLayer
from models.transformers import TransformerDecoder, TransformerDecoderLayer
from models.transformers import NLPTransformer 
from models.transformers import LanguageModelingHead
from transformers import AutoTokenizer

def main():
    dim = 64
    mlp_dim = 128
    n_heads = 8

    attn = MultiHeadAttention(dim, n_heads)
    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=3)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=3)
    tokenizer = AutoTokenizer.from_pretrained('../models/facebook/bart-base', padding_side='right', clean_up_tokenization_spaces=True)
    transformer = NLPTransformer(tokenizer, encoder = encoder, decoder = decoder, padding = True)
    model = LanguageModelingHead(transformer)

if __name__ == 'main':
    main()
