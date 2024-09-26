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

            print('Tensors allocated')

            self.tgt = df[tgt_key].tolist()

            batch_size = 500
            batches = len(self.src) // batch_size

            # TODO: last batch
            if self.tokenizer:
                for i in range(batches):
                    print(f'Processing batch {i}/{batches}')
                    idx_text = slice(0, batch_size)
                    idx_tokens = slice(i*batch_size, (i + 1)*batch_size)

                    src = tokenizer(self.src[idx_text], padding=padding, max_length=max_length, truncation=truncation, return_tensors='np')
                    src_t[idx_tokens] = src.input_ids.copy()
                    src_masks_t[idx_tokens] = src.attention_mask.copy()

                    tgt = tokenizer(self.tgt[idx_text], padding=padding, max_length=max_length, truncation=truncation, return_tensors='np')
                    tgt_t[idx_tokens] = tgt.input_ids.copy()
                    tgt_masks_t[idx_tokens] = tgt.attention_mask.copy()

                    print(src)
                    print(tgt)

                    del src
                    del tgt
                    del self.src[:batch_size]
                    del self.tgt[:batch_size]


                    if i % 100 == 0:
                        gc.collect()
                self.src = src_t
                self.tgt = tgt_t
                self.src_masks = src_masks_t
                self.tgt_masks = tgt_masks_t

            if len(self.src) != len(self.tgt):
                raise ValueError('src and tgt must have the same length')

    def save_dump(self, file: str):
        """ Save dataset as an uncompressed npz file """
        np.savez(file, src=self.src, tgt=self.tgt, src_masks=self.src_masks, tgt_masks=self.tgt_masks)
        # np.savez(file, src=self.src, tgt=self.tgt, src_masks=self.src_masks, tgt_masks=self.tgt_masks)
        # torch.save({'src': self.src_t, 'tgt': self.tgt_t, 'src_masks': self.src_masks_t, 'tgt_masks': self.tgt_masks_t}, file)

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

