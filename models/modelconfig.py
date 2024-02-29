from dataclasses import dataclass, field
from typing import List

@dataclass
class CNNGRUConfig:
    input_size: int
    hidden_size: int
    predict_last: bool
    output_size: int = 1

@dataclass
class TCNConfig:
    input_size: int
    channel_list: list
    kernel_size: int
    dropout: float
    output_size: int = 1