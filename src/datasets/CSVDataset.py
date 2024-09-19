import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from tokenizers import Tokenizer
# from pytorch_transformers.tokenization_utils import PretrainedTokenizer
# TODO correct type annotation for Tokenizer

class CSVDataset(Dataset):
    """
    Construct a torch.utils.data.Dataset from a CSV file with its entries possibly
    tokenized.
    """
    # TODO: pretokenization parameter, online tokenization
    def __init__(
            self,
            csv_path: str,
            src_key: str,
            tgt_key: str = None,
            tokenizer: Tokenizer = None,
            device: torch.device = 'cpu'
            ) -> None:
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.src = df[src_key].tolist()
        self.tgt = []

        self.src_masks = []
        self.tgt_masks = []

        if tgt_key:
            self.tgt = df[tgt_key].tolist()

        if self.tokenizer:
            src = tokenizer(self.src, padding='longest', return_tensors='pt')
            self.src = src.input_ids.to(device)
            self.src_masks = src.attention_mask.to(device)

            if self.tgt:
                tgt = tokenizer(self.tgt, padding='longest', return_tensors='pt')
                self.tgt = tgt.input_ids.to(device)
                self.tgt_masks = tgt.attention_mask.to(device)

        if self.tgt is not [] and len(self.src) != len(self.tgt):
            raise ValueError('src and tgt must have the same length')

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx) -> Tensor:
        entry = (self.src[idx],)
        if self.tgt is not []:
            entry = entry + (self.tgt[idx],)

        if self.tokenizer:
            return entry + (self.src_masks[idx], self.tgt_masks[idx])
        return entry
