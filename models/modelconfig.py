from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    input_size: int
    hidden_size: int
    predict_last: bool
    output_size: int = 1
