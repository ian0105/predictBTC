
import numpy as np
import pytorch_lightning as pl
import torch

from models.CNNGRU import CNNGRU
from models.loss import MSELoss,BinaryCrossEntropyLoss
from models import TCN, CNNGRU
from typing import Union

class CoinPredict(pl.LightningModule):
    def __init__(self,
                 model_type: Union[CNNGRU, TCN],
                 initial_learning_rate: float,
                 loss: str
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_type"])
        self.model = model_type
        if loss == 'mse':
            self.loss = MSELoss()
        elif loss == 'binary':
            self.loss =BinaryCrossEntropyLoss()
        self.validation_step_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(),lr=self.hparams.initial_learning_rate)
        return opt

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
        data, label = batch
        if len(data.shape)==2:
            data = data.unsqueeze(-1)
        output = self(data)
        if len(output.shape)==3:
            output = output.squeeze(-1)
        loss = self.loss(output, label)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        if len(data.shape)==2:
            data = data.unsqueeze(-1)
        output = self(data)

        if len(output.shape)==3:
            output = output.squeeze(-1)
        loss = self.loss(output, label)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = sum(outputs) / len(outputs)
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()