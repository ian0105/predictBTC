import numpy as np
import torch
import torchaudio
import math
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional, List, Dict, Any
from .dataconfig import DataConfig
from tqdm import tqdm


class Base(Dataset):
    def __init__(self, data_list):

        self.datalist = data_list

    def __len__(self) -> int:
        return len(self.datalist)


    def _normalize(self, data):
        min_val = min(data)
        max_val = max(data)

        scaled_data = [(x - min_val) / (max_val - min_val) for x in data]

        return scaled_data

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.datalist[index]
        data = self._normalize(data)
        input = data[:-1]
        label = data[-1]
        return input, label
