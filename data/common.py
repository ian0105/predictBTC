import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional, List, Dict, Any
from .dataconfig import DataConfig
from tqdm import tqdm


class Base(Dataset):
    def __init__(self, data_list):

        self.datalist = data_list

    def __len__(self) -> int:
        return len(self.datalist)



    def __getitem__(self, index: int) -> torch.Tensor:
        input, label = self.datalist[index]

        return input, label
