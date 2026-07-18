import torch
import torch.nn as nn
from src.positional_encoding import PositionalEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, n_aksi=10, n_jeda=22, n_kategori=52, n_harga=12,
        dim=128, n_head=4, n_layers=2, ff=256, dropout=0.1, maxlen=128):
        super().__init__()
        sub_dim = dim // 4
        self.emb_aksi = nn.Embedding(n_aksi, sub_dim, padding_idx=0)
        self.emb_jeda = nn.Embedding(n_jeda, sub_dim, padding_idx=0)
        self.emb_kategori = nn.Embedding(n_kategori, sub_dim, padding_idx=0)
        self.emb_harga = nn.Embedding(n_harga, sub_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoder(maxlen=maxlen, dim=dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_head, dim_feedforward=ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, aksi, jeda, kategori, harga):
        pad_mask = (aksi == 0)

        x = torch.cat([
            self.emb_aksi(aksi),
            self.emb_jeda(jeda),
            self.emb_kategori(kategori),
            self.emb_harga(harga),
        ], dim=-1)
        x = self.pos_encoder.encode(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x, pad_mask


class Classifier(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x, pad_mask):
        mask = (~pad_mask).unsqueeze(-1).float()
        x_mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.linear(x_mean).squeeze(-1)


class DetectorModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim = kwargs.get("dim", 128)
        self.encoder = TransformerEncoder(**kwargs)
        self.classifier = Classifier(dim=dim)

    def forward(self, aksi, jeda, kategori, harga):
        x, pad_mask = self.encoder(aksi, jeda, kategori, harga)
        return self.classifier(x, pad_mask)
