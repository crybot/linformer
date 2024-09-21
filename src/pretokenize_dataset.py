import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from datasets import CSVDataset

# TODO: command line arguments

def main():
    max_length = 200
    tokenizer = AutoTokenizer.from_pretrained(
            './HLT/models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=False
            )

    print('Tokenizing dataset')
    dataset = CSVDataset(
            './HLT/datasets/wmt14_translate_de-en_train.csv',
            src_key = 'en',
            tgt_key='de',
            tokenizer=tokenizer,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            device='cpu'
            )
    print('Tokenization complete')
    print('Saving dataset dump')

    dataset.save_dump("./HLT/datasets/wmt14-tokenized")

    print('Dump successfully saved')

if __name__ == '__main__':
    main()
