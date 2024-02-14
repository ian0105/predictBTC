
import numpy as np
import pytorch_lightning as pl
import torch

from models.CNNGRU import CNNGRU
from models.loss import MSELoss
from models.modelconfig import Config


class CoinPredict(pl.LightningModule):
    def __init__(self,
                 model_config: Config
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNNGRU()
        self.mseloss = MSELoss()

        self.validation_step_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(),lr=self.hparams.initial_learning_rate)
        return opt

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
        data, label = batch
        target = self(data)
        loss = self.mseloss(target,label)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        target = self(data)
        loss = self.mseloss(target, label)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = sum(outputs) / len(outputs)
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()