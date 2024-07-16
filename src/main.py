#!/usr/bin/python3
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from utils import set_random_state
from datasets import CSVDataset


def main():
    SEED = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    ########## TODO ############

    dataset = CSVDataset('./HLT/datasets/bbc-text.csv', 'text')

    ############################

    print('Done')


if __name__ == '__main__':
    main()
