from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, default_collate
from data.dataconfig import DataConfig
from data.upbitdata import Upbitdata


class UpbitDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = Upbitdata(cfg, train=train)
        dataloader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
            )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)