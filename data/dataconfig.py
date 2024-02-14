from dataclasses import dataclass


@dataclass
class DataConfig:
    batch_size: int
    num_workers: int
    filelist_path: str
    data_period: int
    train_ratio: float
    target_data: str
    data_term: int