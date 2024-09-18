import torch

def freeze_(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
