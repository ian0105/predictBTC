import torch
import torch.nn as nn
import torch.nn.functional as F

from .modelconfig import Config

class CNNGRU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        # Conv1d 레이어
        self.conv1d = nn.Conv1d(in_channels=cfg.input_size, out_channels=cfg.hidden_size, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        #self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        # 첫 번째 GRU 레이어
        self.gru1 = nn.GRU(cfg.hidden_size, cfg.hidden_size, batch_first=True)

        # 두 번째 GRU 레이어
        self.gru2 = nn.GRU(cfg.hidden_size, cfg.hidden_size, batch_first=True)

        # Dense 레이어
        self.dense = nn.Linear(cfg.hidden_size, cfg.output_size)

        self.predict_last = cfg.predict_last

    def forward(self, x):
        # (B, L, C) -> (B, C, L)
        x = x.transpose(-2,-1)
        # Conv1d 레이어 적용
        x = self.conv1d(x)
        x = x.transpose(-2, -1)
        x = self.relu(x)
        #x = self.layernorm(x)
        # (B, C, L) -> (B, L, C)

        # GRU 레이어 1 적용
        x, _ = self.gru1(x)

        # GRU 레이어 2 적용
        x, _ = self.gru2(x)

        # 마지막 시간 단계의 출력만 사용
        if self.predict_last:
            x = x[..., -1, :]

        # Dense 레이어 적용
        x = self.dense(x)

        x = torch.sigmoid(x)

        return x
