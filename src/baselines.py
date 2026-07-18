import torch
import torch.nn as nn
from src.model import Classifier

class LSTMEncoder(nn.Module):
    def __init__(self, n_aksi=10, n_jeda=22, n_kategori=52, n_harga=12,
        dim=128, n_layers=1, dropout=0.1, maxlen=128):
        super().__init__()
        sub_dim = dim // 4
        self.emb_aksi = nn.Embedding(n_aksi, sub_dim, padding_idx=0)
        self.emb_jeda = nn.Embedding(n_jeda, sub_dim, padding_idx=0)
        self.emb_kategori = nn.Embedding(n_kategori, sub_dim, padding_idx=0)
        self.emb_harga = nn.Embedding(n_harga, sub_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=n_layers,
            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))

    def forward(self, aksi, jeda, kategori, harga):
        pad_mask = (aksi == 0)
        x = torch.cat([
            self.emb_aksi(aksi),
            self.emb_jeda(jeda),
            self.emb_kategori(kategori),
            self.emb_harga(harga),
        ], dim=-1)
        x, _ = self.lstm(x)
        return x, pad_mask


class GRUEncoder(nn.Module):
    def __init__(self, n_aksi=10, n_jeda=22, n_kategori=52, n_harga=12,
        dim=128, n_layers=1, dropout=0.1, maxlen=128):
        super().__init__()
        sub_dim = dim // 4
        self.emb_aksi = nn.Embedding(n_aksi, sub_dim, padding_idx=0)
        self.emb_jeda = nn.Embedding(n_jeda, sub_dim, padding_idx=0)
        self.emb_kategori = nn.Embedding(n_kategori, sub_dim, padding_idx=0)
        self.emb_harga = nn.Embedding(n_harga, sub_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=n_layers,
            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))

    def forward(self, aksi, jeda, kategori, harga):
        pad_mask = (aksi == 0)
        x = torch.cat([
            self.emb_aksi(aksi),
            self.emb_jeda(jeda),
            self.emb_kategori(kategori),
            self.emb_harga(harga),
        ], dim=-1)
        x, _ = self.gru(x)
        return x, pad_mask


class LSTMDetector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim = kwargs.get("dim", 128)
        self.encoder = LSTMEncoder(**kwargs)
        self.classifier = Classifier(dim=dim)

    def forward(self, aksi, jeda, kategori, harga):
        x, pad_mask = self.encoder(aksi, jeda, kategori, harga)
        return self.classifier(x, pad_mask)


class GRUDetector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim = kwargs.get("dim", 128)
        self.encoder = GRUEncoder(**kwargs)
        self.classifier = Classifier(dim=dim)

    def forward(self, aksi, jeda, kategori, harga):
        x, pad_mask = self.encoder(aksi, jeda, kategori, harga)
        return self.classifier(x, pad_mask)
