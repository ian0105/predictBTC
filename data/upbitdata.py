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

class Upbitdata(Base):
    def __init__(
        self,
        cfg: DataConfig,
        train: bool,
    ):
        path = cfg.filelist_path
        data_csv = pd.read_csv(path)
        data = data_csv[cfg.target_data]

        #Chunk list
        chunking_list = lambda d, size, term: [d[i:i + size] for i in range(0, len(lst), size)]
        coin_list = chunking_list(data,cfg.data_period + 1,cfg.data_term)

        random.shuffle(coin_list)

        train_len = len(coin_list) * cfg.train_ratio
        if train:
            coin_list = coin_list[:-train_len]
        else:
            coin_list = coin_list[-train_len:]

        total_len = len(coin_list)
        mapping = {
            t: (
            coin_list[t])
            for t in range(total_len)
        }

        data_list = []
        for track_id, (coin_list) in tqdm(mapping.items()):
            data_list.append((coin_list))
        super().__init__(data_list, cfg, train)




