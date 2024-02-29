import torch
import torch.nn.functional as F
import torch.nn as nn


class MSELoss(nn.Module):
    def forward(self, pred, tar):
        mse_loss = F.mse_loss(pred, tar)
        return mse_loss

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.BCELoss()
    def forward(self, predictions, targets):
        loss = self.loss_function(predictions, targets.float())
        return loss