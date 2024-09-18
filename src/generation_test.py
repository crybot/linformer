#!/usr/bin/python3
import torch
import numpy as np
import warnings
from models import Summarizer
from datasets import CSVDataset
from text.utils import random_mask, random_tokens_extract, random_text_extract, mask_fill
from utils import set_random_state


def test_generation():
    SEED = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    set_random_state(SEED)

    print("Generation test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summarizer = Summarizer(
                pt_lm_path="./HLT/models/facebook/bart-large-cnn",
                fpt_lm_path="./HLT/models/google-bert/bert-large-uncased",
                ).to(device)

    print("Loading dataset")
    dataset = CSVDataset("./HLT/datasets/bbc-text.csv", "text")

    # summarizer.eval()

    print("Generation examples")
    for epoch in range(1):
        for text in dataset[:1]:

            inputs = summarizer.tokenize(text)
            tokens = inputs["input_ids"]
            generated = summarizer.generate(tokens, device=device)


            print(generated)

            # TODO: decode
            
        print("Success")

test_generation()

