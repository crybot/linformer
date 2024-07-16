import torch
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __tokenize__(self, text: str) -> list[str]:
        pass
