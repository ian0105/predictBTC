import torch
from CNNGRU import CNNGRU
import numpy as np
import unittest


class ModelTestCase(unittest.TestCase):
    BATCH_SIZE = 32
    CH = 64

    @unittest.skip("")
    def test_upsample(self):
        x = torch.randn(self.BATCH_SIZE, self.CH, self.H, self.W)
        net = Upsample(self.CH)
        x = net(x)
        assert x.shape == torch.Size([self.BATCH_SIZE, self.CH, self.H * 2, self.W * 2])

    def test_model(self):
        device = 'cuda'
        x1 = torch.randn(self.BATCH_SIZE, CH)
        model = CNNGRU()
        z1, _, _ = model.encode(x1)



if __name__ == '__main__':
    unittest.main(verbosity=2)