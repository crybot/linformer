import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import gc

# TODO: could be more general
# TODO: use torch.save and torch.load instead
class CSVDataset(Dataset):
    """
    Construct a torch.utils.data.Dataset from a CSV file with its entries possibly
    tokenized.
    """
    # TODO: pretokenization parameter, online tokenization
    # TODO: if from_dump = False, src_key must be not None
    # TODO: tgt is required
    # TODO: change name to EncoderDecoderDataset or similar
    def __init__(
            self,
            path: str,
            from_dump: bool = False,
            src_key: str = None,
            tgt_key: str = None,
            tokenizer: Tokenizer = None,
            padding: str = 'longest',
            max_length: int = None,
            truncation: bool = False,
            device: torch.device = 'cpu'
            ) -> None:
        self.from_dump = from_dump
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.device = device

        if from_dump:
            self._load_dump(path)
        else:
            df = pd.read_csv(path, lineterminator='\n')
            self.src = df[src_key].tolist()
            self.tgt = []

            self.src_masks = []
            self.tgt_masks = []

            src_t = np.zeros((len(self.src), max_length), dtype=int)
            tgt_t = np.zeros((len(self.src), max_length), dtype=int)
            src_masks_t = np.zeros((len(self.src), max_length), dtype=int)
            tgt_masks_t = np.zeros((len(self.src), max_length), dtype=int)

            self.tgt = df[tgt_key].tolist()

            batch_size = 500
            batches = len(self.src) // batch_size

            if len(self.src) != len(self.tgt):
                raise ValueError('src and tgt must have the same length')

            if self.tokenizer:
                i = 0
                while len(self.src) > 0:
                    print(f'Processing batch {i}/{batches}')
                    idx = slice(0, min(batch_size, len(self.src)))
                    src, tgt, src_masks, tgt_masks = self._tokenize_batch(
                        self.src[idx],
                        self.tgt[idx]
                        )

                    idx = slice(i*batch_size, i*batch_size + len(src))
                    src_t[idx] = src
                    src_masks_t[idx] = src_masks
                    tgt_t[idx] = tgt
                    tgt_masks_t[idx] = tgt_masks

                    del self.src[:batch_size]
                    del self.tgt[:batch_size]

                    if i % 100 == 0:
                        gc.collect()

                    i += 1

                self.src = src_t
                self.tgt = tgt_t
                self.src_masks = src_masks_t
                self.tgt_masks = tgt_masks_t


    def _tokenize_batch(self, src, tgt):
        assert len(src) == len(tgt)

        encoded = self.tokenizer(
            src,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors='np'
            )
        src = encoded.input_ids.copy()
        src_masks = encoded.attention_mask.copy()

        encoded = self.tokenizer(
            tgt,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors='np'
            )
        tgt = encoded.input_ids.copy()
        tgt_masks = encoded.attention_mask.copy()

        return src, tgt, src_masks, tgt_masks

    def save_dump(self, file: str):
        """ Save dataset as an uncompressed npz file """
        np.savez(file, src=self.src, tgt=self.tgt, src_masks=self.src_masks, tgt_masks=self.tgt_masks)

    def _load_dump(self, npz_path: str) -> None:
        """ 
        Load already preprocessed dataset from a numpy binary file
        """
        with np.load(npz_path) as nps:
            self.src = torch.from_numpy(nps['src'])
            self.tgt = torch.from_numpy(nps['tgt'])
            self.src_masks = torch.from_numpy(nps['src_masks'])
            self.tgt_masks = torch.from_numpy(nps['tgt_masks'])
        gc.collect()

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx) -> Tensor:
        entry = (self.src[idx],)
        if self.tgt is not []:
            entry = entry + (self.tgt[idx],)

        if self.from_dump or self.tokenizer:
            return entry + (self.src_masks[idx], self.tgt_masks[idx])
        return entry

