import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=1):
        super().__init__()

        # Conv1d 레이어
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        # 첫 번째 GRU 레이어
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # 두 번째 GRU 레이어
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Dense 레이어
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # (B, L, C) -> (B, C, L)
        x = x.transpose(1,2)
        # Conv1d 레이어 적용
        x = self.conv1d(x)
        x = self.relu(x)
        x = F.normalize(x)
        # (B, C, L) -> (B, L, C)
        x = x.transpose(1,2)

        # GRU 레이어 1 적용
        x, _ = self.gru1(x)

        # GRU 레이어 2 적용
        x, _ = self.gru2(x)

        # 마지막 시간 단계의 출력만 사용
        x = x[:, -1, :]

        # Dense 레이어 적용
        x = self.dense(x)

        x = torch.sigmoid(x, dim=-1)

        return x