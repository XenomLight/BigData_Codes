# Import library yang dibutuhkan
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import time
import kagglehub
import os

path = kagglehub.dataset_download("lorentzyeung/price-paid-data-202304")
csv_file = os.path.join(path, "pp-complete.csv")


columns = [
    'Price', 'Date_of_Transfer', 'Postcode', 'Property_Type',
    'Old/New', 'Duration', 'PAON', 'SAON', 'Street',
    'Locality', 'Town/City', 'District', 'County',
    'PPD_Category_Type', 'Record_Status'
]

# Gunakan usecols jika hanya ingin kolom tertentu, atau pastikan names sesuai jumlah kolom di file
# 1. Tentukan hanya kolom yang benar-benar memberikan nilai analisis
essential_cols = [
    'Price', 'Date_of_Transfer', 'Property_Type',
    'Old/New', 'Duration', 'Town/City', 'District', 'County'
]

# 2. Tentukan tipe data kategori untuk menghemat RAM (sangat efektif!)
dtype_opt = {
    'Property_Type': 'category',
    'Old/New': 'category',
    'Duration': 'category',
    'Town/City': 'category',
    'District': 'category',
    'County': 'category'
}

# 3. Load data (tanpa PAON, SAON, Street, Record_Status)
df = pd.read_csv(
    csv_file,
    names=columns,
    header=0,
    usecols=essential_cols,
    dtype=dtype_opt
)

print(df.head())
df.info()
print("1. Mulai tahap Data Pre-processing...")

# Pastikan kolom tanggal berformat Datetime
df['Date_of_Transfer'] = pd.to_datetime(df['Date_of_Transfer'])
df = df.dropna()

print("\n2. Encoding variabel kategorik...")
kolom_kategori = ['Property_Type', 'Old/New', 'Duration', 'Town/City', 'District', 'County']
le = LabelEncoder()
for col in kolom_kategori:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Ekstrak Tahun sebagai fitur numerik
df['Year'] = df['Date_of_Transfer'].dt.year

print("\n3. Membagi Data Berdasarkan Waktu (Temporal Split)...")
tahun_terakhir = df['Date_of_Transfer'].dt.year.max()
batas_awal_testing = pd.to_datetime(f"{tahun_terakhir}-04-01") # Pemotongan 1 April 2026

print(f"Memotong data dengan batas waktu: {batas_awal_testing.date()}")
train_data = df[df['Date_of_Transfer'] < batas_awal_testing].copy()
test_data = df[df['Date_of_Transfer'] >= batas_awal_testing].copy()

# Hapus kolom tanggal asli
train_data = train_data.drop(columns=['Date_of_Transfer'])
test_data = test_data.drop(columns=['Date_of_Transfer'])

X_train = train_data.drop(columns=['Price'])
y_train = train_data['Price']
X_test = test_data.drop(columns=['Price'])
y_test = test_data['Price']

print("\n4. Mulai melatih model (Tanpa Scaling)...")
rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=4)
xgb_model = XGBRegressor(n_estimators=50, max_depth=10, tree_method='hist', random_state=42, n_jobs=4)

start_time = time.time()
rf_model.fit(X_train, y_train)
print(f"Random Forest selesai dilatih dalam {time.time() - start_time:.2f} detik")

start_time = time.time()
xgb_model.fit(X_train, y_train)
print(f"XGBoost selesai dilatih dalam {time.time() - start_time:.2f} detik")

print("\n5. Menghitung prediksi pada data testing (Bulan April)...")
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
final_pred = (rf_pred + xgb_pred) / 2 # Ensemble (Rata-rata)

print("\n6. EVALUASI MODEL LENGKAP...")
# Menghitung semua metrik
rmse = np.sqrt(mean_squared_error(y_test, final_pred))
mae = mean_absolute_error(y_test, final_pred)
mape = mean_absolute_percentage_error(y_test, final_pred) * 100 # Dikali 100 agar jadi persen
r2 = r2_score(y_test, final_pred)

print("="*50)
print(f"HASIL METRIK (Data April {tahun_terakhir}):")
print(f"- Rata-rata Harga Asli : £{y_test.mean():,.2f}")
print(f"- RMSE (Error Kuadrat) : £{rmse:,.2f}")
print(f"- MAE (Error Mutlak)   : £{mae:,.2f}")
print(f"- MAPE (Persen Error)  : {mape:.2f}% (Semakin mendekati 0% makin baik)")
print(f"- R² (Tingkat Akurasi) : {r2:.4f} (Semakin mendekati 1.0 makin baik)")
print("="*50)

print("\n7. ANALISIS FEATURE IMPORTANCE (Faktor Penentu Harga)...")
# Mengambil nilai kepentingan fitur dari kedua model
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Membuat tabel peringkat fitur
feature_df = pd.DataFrame({
    'Nama Fitur': X_train.columns,
    'Skor RF': rf_importance,
    'Skor XGB': xgb_importance
})
# Menghitung rata-rata skor kepentingan dari kedua model
feature_df['Skor Gabungan'] = (feature_df['Skor RF'] + feature_df['Skor XGB']) / 2
feature_df = feature_df.sort_values(by='Skor Gabungan', ascending=False)

print("Peringkat Fitur Paling Berpengaruh terhadap Harga Rumah:")
print(feature_df[['Nama Fitur', 'Skor Gabungan']].to_string(index=False))

print("\n8. Perbandingan Harga Asli vs Prediksi (10 Data Terakhir)...")
comparison_df = pd.DataFrame({
    'Harga Asli (£)': y_test.values,
    'Prediksi RF (£)': rf_pred,
    'Prediksi XGB (£)': xgb_pred,
    'Prediksi Gabungan (£)': final_pred,
    'Selisih Error (£)': np.abs(y_test.values - final_pred)
})
last_10_comparison = comparison_df.tail(10)
# Memperpanjang garis pemisah agar tabel yang lebih lebar terlihat rapi
print("-" * 90)
print(last_10_comparison.to_string(index=False, float_format=lambda x: "{:,.2f}".format(x)))
print("-" * 90)