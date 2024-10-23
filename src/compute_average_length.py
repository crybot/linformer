import pandas as pd
from datasets import CSVDataset
from statistics import fmean
from transformers import AutoTokenizer
from time import sleep

def main():
    tokenizer = AutoTokenizer.from_pretrained(
            './HLT/models/facebook/bart-base',
            clean_up_tokenization_spaces=True,
            use_fast=True
            )

    df = pd.read_csv('./HLT/datasets/wmt14_translate_de-en_train.csv', lineterminator='\n')
    src_list = df['en'].tolist()
    tgt_list = df['de'].tolist()
    n = len(src_list)

    src_lengths = []
    tgt_lengths = []
    batch_size = 250
    batches = n // batch_size

    i = 0
    while len(src_list) > 0:
        print(f'Processing batch {i}/{batches}')
        src = src_list[:batch_size]
        tgt = tgt_list[:batch_size]

        encoded = tokenizer(src, padding=False, return_tensors='np')
        src_lengths.extend(map(len, encoded.input_ids))

        encoded = tokenizer(tgt, padding=False, return_tensors='np')
        tgt_lengths.extend(map(len, encoded.input_ids))

        del src_list[:batch_size]
        del tgt_list[:batch_size]
        i += 1

    print(f'Average src length: {fmean(src_lengths)}')
    print(f'Min src length: {min(src_lengths)}')
    print(f'Max src length: {max(src_lengths)}')

    print(f'Average tgt length: {fmean(tgt_lengths)}')
    print(f'Min tgt length: {min(tgt_lengths)}')
    print(f'Max tgt length: {max(tgt_lengths)}')

if __name__ == '__main__':
    main()
