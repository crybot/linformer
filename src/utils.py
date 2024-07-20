import torch
import random
import numpy as np

#TODO: refactor
#TODO: move to appropriate modules
#TODO: type annotations

def set_random_state(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model, print_model=False):
    if print_model:
        print(model)
        print(f'Number of parameters: {params_count(model)}')


