import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def _build_events(session_id, n_events, delays, catalog):
    kategori_terpilih = catalog["category_code"].sample(n=1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
    kandidat_fokus = catalog[catalog["category_code"] == kategori_terpilih]
    if len(kandidat_fokus) == 0:
        kandidat_fokus = catalog

    baris_item = []
    for _ in range(n_events):
        if RNG.random() < 0.08:
            sumber = catalog
        else:
            sumber = kandidat_fokus
        baris_item.append(sumber.sample(n=1, random_state=int(RNG.integers(0, 1_000_000))))
    items = pd.concat(baris_item, ignore_index=True)

    event_types = ["view"] * (n_events - 2) + ["cart", "purchase"]
    event_types = event_types[:n_events]

    start = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=int(RNG.integers(0, 86400)))
    times = [start]
    for d in delays[: n_events - 1]:
        times.append(times[-1] + pd.Timedelta(seconds=float(d)))

    return pd.DataFrame({
        "event_time": times,
        "event_type": event_types,
        "product_id": items["product_id"],
        "category_code": items["category_code"],
        "brand": items["brand"],
        "price": items["price"],
        "user_session": session_id,
    })


def naive_bot_session(session_id, n_events=20, interval=0.4, catalog=None):
    delays = np.full(n_events - 1, interval)
    return _build_events(session_id, n_events, delays, catalog)


def jitter_bot_session(session_id, n_events=20, base_interval=0.6, jitter_std=0.15, catalog=None):
    delays = RNG.normal(base_interval, jitter_std, size=n_events - 1)
    delays = np.clip(delays, 0.05, None)
    return _build_events(session_id, n_events, delays, catalog)


def adaptive_bot_session(session_id, n_events, human_delays, catalog=None, decoy_prob=0.08):
    delays = RNG.choice(human_delays, size=n_events - 1, replace=True)
    df = _build_events(session_id, n_events, delays, catalog)
    for i in range(2, len(df)):
        if RNG.random() < decoy_prob:
            df.loc[i, ["product_id", "category_code", "brand", "price"]] = df.loc[i - 2, ["product_id", "category_code", "brand", "price"]].values
            df.loc[i, "event_type"] = "view"
    return df
