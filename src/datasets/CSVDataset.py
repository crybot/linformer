import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from text.tokenizers import Tokenizer
# from pytorch_transformers.tokenization_utils import PretrainedTokenizer
# TODO correct type annotation for Tokenizer

class CSVDataset(Dataset):
    """
    Construct a torch.utils.data.Dataset from a CSV file with its entries possibly
    tokenized.
    """
    def __init__(self, csv_path: str, text_key: str, features_keys: list[str] = [], tokenizer: Tokenizer = None) -> None:
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

        if self.tokenizer:
            self.dataset = [tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                return_tensors="pt") for text in df['text']]
        else:
            self.dataset = [text for text in df[text_key]]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tensor:
        return self.dataset[idx]
