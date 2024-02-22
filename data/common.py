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


    def _normalize(tensor, min_value=None, max_value=None):
        # 최솟값과 최댓값을 이용하여 정규화
        if min_value:
            scaled_tensor = (tensor - min_value) / (max_value - min_value)
        else:
            scaled_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return scaled_tensor, tensor.min(), tensor.max()

    def _unnormalize(scaled_tensor, original_min, original_max):
        # 원래 범위로 되돌리는 역변환 수행
        reversed_tensor = scaled_tensor * (original_max - original_min) + original_min

        return reversed_tensor
    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.datalist[index]
        input = torch.FloatTensor(data[:-1])
        input, min_value, max_value = self._normalize(input)
        label = torch.FloatTensor([data[-1]])
        label = self._normalize(label, min_value, max_value)
        return input, label
