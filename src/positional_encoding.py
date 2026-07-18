import torch
import numpy as np

class PositionalEncoder:
    def __init__(self, maxlen=128, dim=128):
        self.maxlen = maxlen
        self.dim = dim
        self.pe = self._build()

    def _build(self):
        position = torch.arange(self.maxlen).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-np.log(10000.0) / self.dim))
        pe = torch.zeros(self.maxlen, self.dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def encode(self, x):
        return x + self.pe.unsqueeze(0).to(x.device)
