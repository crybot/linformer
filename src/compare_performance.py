import os
import torch
import wandb
import argparse
import numpy as np
from torch import nn, Tensor
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import CSVDataset
from evaluation.metrics import perplexity
from evaluation.utils import extract_probs
from utils import to_device, print_summary, set_random_state, download_wandb_checkpoint
from utils import make_model, download_wandb_config, load_model_from_wandb_checkpoint
from utils import load_config
from sacrebleu import corpus_bleu
from text.utils import encoder_decoder_inputs, trim_batch_pad_tokens
from timeit import timeit

def random_data(batch_size, length, device = 'cpu') -> Tensor:
    src = torch.randint(0, 50000, (batch_size, length), device=device)
    return src, src.clone()

def inference(model, batch_size, length, device = 'cpu'):
    src, tgt = random_data(batch_size, length, device=device)
    model(src, tgt)

def main(args):
    device = 'cuda'
    max_tokens = 256 * 64
    lengths = [2 ** n for n in range(8, 15)] # starting from 256

    config = load_config(os.path.join('./HLT', args.config))

    with torch.no_grad(), torch.autocast(device_type=device):
        for length in lengths:
            batch_size = max_tokens // length
            model = make_model(config, device=device)
            model.eval()

            inference(model, batch_size, length, device=device) # warmup

            time = timeit(lambda: inference(model, batch_size, length, device=device), number = 10)

            print(f'n = {length}, batch size = {batch_size}')
            print(f'Elapsed (s): {time}')
            print(f'torch.cuda.memory_allocated (GB): {torch.cuda.memory_allocated(0)/2**30}')
            print(f'torch.cuda.memory_reserved (GB): {torch.cuda.memory_reserved(0)/2**30}')
            print(f'torch.cuda.max_memory_reserved (GB): {torch.cuda.max_memory_reserved(0)/2**30}')
            print()
            torch.cuda.empty_cache()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.config:
        raise ValueError("No config file provided")
    main(args)
