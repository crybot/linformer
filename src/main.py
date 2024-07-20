#!/usr/bin/python3
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from utils import set_random_state
from datasets import CSVDataset
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from text.utils import random_mask, random_extract, mask_fill


def main():
    SEED = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    # set_random_state(SEED)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    ########## TODO ############

    # Note: bbc-text is missing punctuations and other special characters
    dataset = CSVDataset('./HLT/datasets/bbc-text.csv', 'text')

    # Pretrained LM fine-tuned on summarization
    pt_lm = AutoModelForSeq2SeqLM.from_pretrained(
            "./HLT/models/facebook/bart-large-cnn"
            ).to(device)

    # Pretrained LM with frozen weights
    fpt_lm = AutoModelForMaskedLM.from_pretrained(
            "./HLT/models/facebook/bart-large",
            forced_bos_token_id=0
            ).to(device)

    # can share tokenizer between pt_lm and fpt_lm
    tokenizer = AutoTokenizer.from_pretrained("./HLT/models/facebook/bart-large-cnn", padding_side="right")

    sentences = ["What do you think are the main reasons we exist in this world?", "Hello this is just a simple sentence.", "Oh no, another sentence"]
    input_ids = tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"].to(device)
    masked_ids, masked_index = random_mask(input_ids, tokenizer.mask_token_id, mask_p=0.2, pad_token_id=tokenizer.pad_token_id)

    print(f'Original sentences:')
    for i, s in enumerate(sentences):
        print(f'Sentence {i+1}: {s}')

    masked_decoded = tokenizer.batch_decode(
            masked_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
            )

    print(f'Masked sentences:')
    for i, s in enumerate(masked_decoded):
        print(f'Sentence {i+1}: {s}')

    predictions = mask_fill(fpt_lm, masked_ids, masked_index, top_k = 5)

    for i, pred in enumerate(predictions):
        print(f'Sentence {i+1} predictions: {[tokenizer.batch_decode(p) for p in pred]}')

    ############################

    print('Done')


if __name__ == '__main__':
    main()
