import torch
import torch.nn.functional as F
import torch.nn as nn


class MSELoss(nn.Module):
    def forward(self, pred, tar):
        mse_loss = F.mse_loss(pred, tar)
        return mse_loss