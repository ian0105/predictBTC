import torch
from CNNGRU import CNNGRU
import numpy as np
import unittest


class ModelTestCase(unittest.TestCase):
    BATCH_SIZE = 32
    CH = 64
    H = 80
    W = 40

    @unittest.skip("")
    def test_upsample(self):
        x = torch.randn(self.BATCH_SIZE, self.CH, self.H, self.W)
        net = Upsample(self.CH)
        x = net(x)
        assert x.shape == torch.Size([self.BATCH_SIZE, self.CH, self.H * 2, self.W * 2])

    def test_vaemodel(self):
        device = 'cuda'
        x1 = torch.randn(self.BATCH_SIZE, 1, n_mels, 40)
        vae = VAEModel(1, 128)
        z1, _, _ = vae.encode(x1)
        z2, _, _ = vae.encode(x2)
        assert z1.shape == z2.shape

        new_z = interpolate_latent_vectors(z1, z2, 0.5)
        assert new_z.shape == z1.shape

        out = vae.inference(new_z)
        assert out.shape == x1.shape


if __name__ == '__main__':
    unittest.main(verbosity=2)