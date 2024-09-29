import torch
import random
import numpy as np
import wandb
import os
from torch import nn, Tensor
import yaml
from models.transformers import Transformer, MultiHeadAttention, LinformerAttention
from models.transformers import TransformerEncoder, TransformerEncoderLayer
from models.transformers import TransformerDecoder, TransformerDecoderLayer
from models.transformers import NLPTransformer 
from models.transformers import LanguageModelingHead

#TODO: refactor
#TODO: move to appropriate modules
#TODO: type annotations

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model, print_model=False):
    if print_model:
        print(model)
    print(f'Number of parameters: {params_count(model)}')

def to_device(*tensors, device):
    return tuple(tensor.to(device) for tensor in tensors)

def make_model(config: dict, device='cpu') -> nn.Module:
    n = config['dataset']['max_length']
    config = config['model']
    dim = config['dim']
    mlp_dim = config['mlp_dim']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    vocab_size = config['vocab_size']

    if config.get('type', None) == 'Linformer':
        attn = LinformerAttention(dim, n_heads, k = config['k'], sequence_length = n)
    else:
        attn = MultiHeadAttention(dim, n_heads)

    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=n_layers)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=n_layers)

    transformer = NLPTransformer(encoder = encoder, decoder = decoder, vocab_size = vocab_size)
    return LanguageModelingHead(transformer).to(device)

def download_wandb_checkpoint(run_path, filename, device='cuda', **kwargs):
    api = wandb.Api()
    run = api.run(run_path)
    run.file(filename).download(**kwargs)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    return checkpoint

def download_wandb_config(run_path, filename, strip_values=True, **kwargs):
    api = wandb.Api()
    run = api.run(run_path)
    run.file(filename).download(**kwargs)
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

    if strip_values:
        config = strip_wandb_values(config)

    return config

def save_wandb_file(path):
    wandb.save(path, base_path=os.path.dirname(path))

def strip_wandb_values(config_dict):
    def recursive_strip(d):
        if isinstance(d, dict) and 'value' in d and len(d) == 1:
            return recursive_strip(d['value'])  # If only 'value' exists, strip it
        elif isinstance(d, dict):
            return {k: recursive_strip(v) for k, v in d.items()}
        else:
            return d
    
    return recursive_strip(config_dict)

def load_model_from_wandb_checkpoint(
        run_path: str,
        checkpoint_path: str = 'checkpoint.pt',
        config_path: str = 'config.yaml',
        device: str = 'cpu'
        ) -> nn.Module:
    checkpoint = download_wandb_checkpoint(run_path, checkpoint_path, device=device, exist_ok = True)
    config = download_wandb_config(run_path, config_path, strip_values=True, replace = True)
    model = make_model(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
