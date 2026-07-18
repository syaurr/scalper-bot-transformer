import pandas as pd

def ekstrak_fitur_sesi(df_sesi):
    durasi = (df_sesi['event_time'].max() - df_sesi['event_time'].min()).total_seconds()
    return {
        'jumlah_event': len(df_sesi),
        'rata_jeda': df_sesi['jeda_detik'].mean(),
        'std_jeda': df_sesi['jeda_detik'].std(),
        'min_jeda': df_sesi['jeda_detik'].min(),
        'maks_jeda': df_sesi['jeda_detik'].max(),
        'jumlah_kategori_unik': df_sesi['category_code'].nunique(),
        'rata_harga': df_sesi['price'].mean(),
        'std_harga': df_sesi['price'].std(),
        'rasio_view': (df_sesi['event_type'] == 'view').mean(),
        'rasio_cart': (df_sesi['event_type'] == 'cart').mean(),
        'rasio_purchase': (df_sesi['event_type'] == 'purchase').mean(),
        'durasi_sesi': durasi,
    }

def bangun_matriks_fitur(df_all):
    baris_fitur = []
    for id_sesi, df_sesi in df_all.groupby('user_session'):
        fitur = ekstrak_fitur_sesi(df_sesi)
        fitur['user_session'] = id_sesi
        fitur['label'] = 0 if df_sesi['label'].iloc[0] == 'human' else 1
        baris_fitur.append(fitur)
    hasil = pd.DataFrame(baris_fitur).fillna(0)
    return hasil
