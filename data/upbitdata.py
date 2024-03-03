from pathlib import Path
import json
from tqdm import tqdm
from .common import Base
import numpy as np
import pandas as pd
import os
import random
from .dataconfig import DataConfig
import torch
import math


class Upbitdata(Base):
    def __init__(
        self,
        cfg: DataConfig,
        train: bool,
    ):
        path = cfg.filelist_path
        data_csv = pd.read_csv(path)
        data = data_csv[cfg.target_data].values
        self.data = data

        #Chunk list
        chunking_list = lambda d, size, term: [d[i:i + size] for i in range(0, len(d), term) if len(d[i:i + size])==size]
        coin_list = chunking_list(data,cfg.data_period + cfg.predict_point,cfg.data_term)
        coin_list = self.preprocess_end_points(coin_list,cfg.data_period + cfg.predict_point,cfg.data_term)

        random.shuffle(coin_list)

        train_len = int(len(coin_list) * cfg.train_ratio)
        if train:
            coin_list = coin_list[:train_len]
        else:
            coin_list = coin_list[train_len:]

        total_len = len(coin_list)
        mapping = {
            t: (
            coin_list[t])
            for t in range(total_len)
        }

        data_list = []
        for track_id, (coin) in tqdm(mapping.items()):
            if cfg.label_format == 'continuous_value':
                input = torch.FloatTensor(coin[:-cfg.predict_point])
                input, min_value, max_value = self._normalize(input)
                label = torch.FloatTensor(coin[cfg.predict_point:])
                # label = torch.FloatTensor([coin[-1]])
                label, _, _ = self._normalize(label, min_value, max_value)
            elif cfg.label_format == 'last_value':
                input = torch.FloatTensor(coin[:-cfg.predict_point])
                input, min_value, max_value = self._normalize(input)
                label = torch.FloatTensor([coin[-1]])
                label, _, _ = self._normalize(label, min_value, max_value)
            elif cfg.label_format == 'up_down':
                input = torch.FloatTensor(coin[:-cfg.predict_point])
                input, min_value, max_value = self._normalize(input)
                if coin[-1] > coin[-cfg.predict_point-1]:
                    label = torch.ones(1)
                else:
                    label = torch.zeros(1)
            data_list.append((input,label))
        super().__init__(data_list)#, cfg, train)
    def _normalize(self, tensor, min_value=None, max_value=None):
        # 최솟값과 최댓값을 이용하여 정규화
        if min_value is not None:
            scaled_tensor = (tensor - min_value) / (max_value - min_value)
        else:
            scaled_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return scaled_tensor, tensor.min(), tensor.max()

    def _unnormalize(self, scaled_tensor, original_min, original_max):
        # 원래 범위로 되돌리는 역변환 수행
        reversed_tensor = scaled_tensor * (original_max - original_min) + original_min

        return reversed_tensor
    def preprocess_end_points(self, coin_list, period, term):
        coin_list[-1] = self.data[-period:]
        return coin_list
        # preprocess last ones of which length is not cfg.data_peroid+1
        #leak_idx = math.ceil(period / term)
        #if coin_list[-leak_idx][-1] == self.data[-1] and len(coin_list[-leak_idx]) == period:
        #    return coin_list[:-leak_idx+1]
        #start = -period
        #end = len(self.data)



        #for i in range(1, leak_idx + 1):
        #    coin_list[-i] = self.data[start:end]
        #    start = -period * (i+1)
        #    end = -period * i
        #return coin_list


