import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from datasets import CSVDataset

# TODO: command line arguments

def main():
    max_length = 256
    fraction = 0.5
    tokenizer = AutoTokenizer.from_pretrained(
            './models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=True
            )

    print('Tokenizing dataset')
    dataset = CSVDataset(
            './datasets/wmt14_translate_de-en_train.csv',
            src_key = 'en',
            tgt_key='de',
            tokenizer=tokenizer,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            fraction=fraction,
            device='cpu'
            )
    print('Tokenization complete')
    print('Saving dataset dump')

    dataset.save_dump(f'./datasets/wmt14-050-tokenized-{max_length}')

    print('Dump successfully saved')

if __name__ == '__main__':
    main()
