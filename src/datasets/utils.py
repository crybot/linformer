import torch
from torch.utils.data import random_split, Sampler, Dataset
from torch.utils.data.dataset import Subset
from typing import Sized, Iterator
import numpy as np
import random
from torch import default_generator, randperm
import itertools

def memory_safe_random_split(dataset, lengths, generator=default_generator, random=True):
    """ same as torch.utils.data.random_split, but passes a `ndarray` to Subset
    in order to avoid memory leaks caused by python refcounting behaviour of
    native lists
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    if random:
        indices = np.array(randperm(sum(lengths), generator=generator).tolist())
    else:
        indices = np.arange(sum(lengths))
        
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(itertools.accumulate(lengths), lengths)]

def split_dataset(ds, train_p=0.7, val_p=0.15, test_p=0.15, random=True, seed=42):
    """ Split a torch.utils.data.Datset into 3 random split, according to the
        given percentages.

        train_p, val_p, test_p must sum up to 1

        returns (train_ds, val_ds, test_ds)
    """
    train_size = int(len(ds) * train_p)

    val_size = int(len(ds) * val_p)
    test_size = len(ds) - train_size - val_size

    train_ds, val_ds, test_ds = memory_safe_random_split(ds,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
            random=random
            )

    # for clarity
    return train_ds, val_ds, test_ds

class RandomSubsetSampler(Sampler[int]):
    def __init__(self, data_source: Sized, fraction: int, replace=False) -> None:
        self.fraction = fraction
        self.data_source = data_source
        self.replace = replace

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if not self.replace:
            yield from iter(np.random.choice(n, int(n*self.fraction), replace=False))
        else:
            for i in range(int(n * self.fraction)):
                yield np.random.randint(low=0, high=n)

    def __len__(self) -> int:
        return int(len(self.data_source) * self.fraction)

class UnbatchedDataloader():
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    def __iter__(self):
        self.consumed = False
        return self

    def __next__(self):
        if self.consumed:
            raise StopIteration
        self.consumed = True
        return self.dataset[:]

    def __len__(self):
        return min(1, len(self.dataset))
