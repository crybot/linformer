import torch
from torch import Tensor, nn
import numpy as np
import pandas as pd
import readline
import os
import atexit
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

