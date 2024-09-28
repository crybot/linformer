import torch
import random
import numpy as np
import wandb
import os

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

def download_wandb_checkpoint(run_path, filename, device='cuda'):
    api = wandb.Api()

    run = api.run(run_path)
    run.file(filename).download(replace=True)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    return checkpoint

def save_wandb_file(path):
    wandb.save(path, base_path=os.path.dirname(path))
