import importlib
import torch
from torch import Tensor, nn
import numpy as np
import pandas as pd
import readline
import os
import atexit
from models.transformers import Transformer, MultiHeadAttention
from models.transformers import TransformerEncoder, TransformerEncoderLayer
from models.transformers import TransformerDecoder, TransformerDecoderLayer
from models.transformers import NLPTransformer 
from models.transformers import LanguageModelingHead
from transformers import AutoTokenizer
import datasets
from datasets import CSVDataset


print("Python interactive shell started with default imports!")

# Point to the history file in the mounted volume
histfile = os.path.join("/mnt/history", ".python_history")

# Load the history if the file exists
try:
    readline.read_history_file(histfile)
except FileNotFoundError:
    pass

# Save the history on exit
atexit.register(readline.write_history_file, histfile)

