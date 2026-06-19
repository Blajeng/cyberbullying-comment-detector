import pandas as pd

print("⏳ Memulai proses penyatuan Master Dataset Lintas Platform...")

# 1. Load Dataset Hasil Kombinasi (Kaggle + HF + GitHub Rizalespe)
df_comb = pd.read_csv('combined_dataset.csv')
df_comb_clean = pd.DataFrame({
    'text': df_comb['String'],
    'label': df_comb['encoded_label'].map({0.0: 'negative', 1.0: 'positive'})
})

# 2. Load Dataset TikTok
df_tk = pd.read_csv('Dataset-Research.csv')
df_tk_clean = pd.DataFrame({
    'text': df_tk['comment'],
    'label': df_tk['sentiment'].map({-1: 'negative', 1: 'positive'})
})

# 3. Penggabungan Raksasa
df_master = pd.concat([df_comb_clean, df_tk_clean], ignore_index=True)

# 4. Data Cleaning: Hapus nilai kosong & buang komentar yang duplikat
df_master = df_master.dropna()
df_master = df_master.drop_duplicates(subset=['text'])

# 5. Eksport ke CSV 
df_master.to_csv('dataset_gabungan.csv', index=False)

print("\n=========================================")
print("DATASET BERHASIL DIGABUNGKAN!")
print("=========================================")
print(f"Total Baris Data Gabungan : {len(df_master)} baris")
print(f"Sentimen Cyberbullying (Negatif): {len(df_master[df_master['label']=='negative'])} baris")
print(f"Sentimen Aman (Positif)        : {len(df_master[df_master['label']=='positive'])} baris")
print("=========================================")