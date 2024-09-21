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

        if from_dump:
            self._load_dump(path)
        else:
            df = pd.read_csv(path, lineterminator='\n')
            self.src = df[src_key].tolist()
            self.tgt = []

            self.src_masks = []
            self.tgt_masks = []

            if tgt_key:
                self.tgt = df[tgt_key].tolist()

            if self.tokenizer:
                src = tokenizer(self.src, padding=padding, max_length=max_length, truncation=truncation, return_tensors='pt')
                self.src = src.input_ids.to(device)
                self.src_masks = src.attention_mask.to(device)

                if self.tgt:
                    tgt = tokenizer(self.tgt, padding=padding, max_length=max_length, truncation=truncation, return_tensors='pt')
                    self.tgt = tgt.input_ids.to(device)
                    self.tgt_masks = tgt.attention_mask.to(device)

            if self.tgt is not [] and len(self.src) != len(self.tgt):
                raise ValueError('src and tgt must have the same length')

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

