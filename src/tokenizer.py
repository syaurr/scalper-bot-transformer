import numpy as np
import pandas as pd

MAXLEN = 128

class Tokenizer:
    def __init__(self, n_kategori=50, n_bin_jeda=20, n_bin_harga=10):
        self.n_kategori = n_kategori
        self.n_bin_jeda = n_bin_jeda
        self.n_bin_harga = n_bin_harga
        self.vocab_aksi = {}
        self.vocab_kategori = {}
        self.bin_jeda = None
        self.bin_harga = None

    def fit(self, df):
        aksi_unik = sorted(df["event_type"].dropna().unique())
        self.vocab_aksi = {v: i + 1 for i, v in enumerate(aksi_unik)}

        top_kategori = df["category_code"].value_counts().head(self.n_kategori - 1).index
        self.vocab_kategori = {v: i + 1 for i, v in enumerate(top_kategori)}

        jeda_norm = np.log1p(df["jeda_detik"].fillna(0).clip(lower=0))
        self.bin_jeda = np.quantile(jeda_norm, np.linspace(0, 1, self.n_bin_jeda + 1))

        harga_valid = df["price"].fillna(df["price"].median())
        self.bin_harga = np.quantile(harga_valid, np.linspace(0, 1, self.n_bin_harga + 1))
        return self

    def _encode_kategori(self, value):
        return self.vocab_kategori.get(value, self.n_kategori)

    def encode_session(self, df_sesi, maxlen=MAXLEN):
        df_sesi = df_sesi.sort_values("event_time")

        aksi_ids = df_sesi["event_type"].map(self.vocab_aksi).fillna(0).astype(int).to_numpy()
        kategori_ids = df_sesi["category_code"].map(self._encode_kategori).astype(int).to_numpy()

        jeda_norm = np.log1p(df_sesi["jeda_detik"].fillna(0).clip(lower=0))
        jeda_ids = np.digitize(jeda_norm, self.bin_jeda[1:-1]) + 1

        harga_ids = np.digitize(df_sesi["price"].fillna(0), self.bin_harga[1:-1]) + 1

        posisi_ids = np.arange(1, len(df_sesi) + 1)

        def pad_potong(arr):
            arr = np.asarray(arr)[:maxlen]
            if len(arr) < maxlen:
                arr = np.pad(arr, (0, maxlen - len(arr)))
            return arr

        return {
            "jenis_aksi": pad_potong(aksi_ids),
            "jeda_waktu": pad_potong(jeda_ids),
            "kategori_produk": pad_potong(kategori_ids),
            "tingkat_harga": pad_potong(harga_ids),
            "posisi": pad_potong(posisi_ids),
        }
