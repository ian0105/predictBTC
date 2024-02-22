import soundfile as sf
from pathlib import Path
import json
from tqdm import tqdm
from .common import Base
import numpy as np
import pandas as pd
import os
import random
from .dataconfig import DataConfig
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
        chunking_list = lambda d, size, term: [d[i:i + size] for i in range(0, len(d), term)]
        coin_list = chunking_list(data,cfg.data_period + 1,cfg.data_term)
        coin_list = self.preprocess_end_points(coin_list,cfg.data_period + 1,cfg.data_term)

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
        for track_id, (coin_list) in tqdm(mapping.items()):
            data_list.append((coin_list))
        super().__init__(data_list)#, cfg, train)
    def preprocess_end_points(self, coin_list, period, term):
        # preprocess last ones of which length is not cfg.data_peroid+1
        leak_idx = math.ceil(period / term)
        if coin_list[-leak_idx][-1] == self.data[-1] and len(coin_list[-leak_idx]) == period:
            return coin_list[:-leak_idx+1]
        start = -period
        end = len(self.data)
        for i in range(1, leak_idx + 1):
            coin_list[-i] = self.data[start:end]
            start = -period * (i+1)
            end = -period * i
        return coin_list


