
Deskripsi
---------
Baseline model ini menggunakan dua algoritma ensemble untuk prediksi harga:
- Random Forest (RF)
- XGBoost (XGB)

Konfigurasi Baseline
--------------------
Model menggunakan dua algoritma ensemble yang dilatih dengan konfigurasi hyperparameter sebagai berikut:

| Model         | n_estimators | max_depth | tree_method | random_state | n_jobs |
| ------------- | ------------ | --------- | ----------- | ------------ | ------ |
| Random Forest | 30           | 10        | —           | 42           | 4      |
| XGBoost       | 50           | 10        | hist        | 42           | 4      |

Variasi Percobaan Model
--------------------
- **Baseline_7SelectedFeatures**: <br> Menggunakan fitur terpilih hasil seleksi peneliti

- **Baseline_AllFeatures**: <br> Menggunakan semua kolom asli setelah preprocessing/encoding 

- **Baseline_Top12ReliefFFeature**: <br> Fitur yang memiliki skor tertinggi hasil dari ReliefF dan merupakan 12 teratas


Catatan singkat
---------------
- Variabel kategorikal di-encoding dengan `LabelEncoder` sebelum pelatihan.
- Evaluasi menggunakan metrik RMSE, MAE, MAPE, dan R² pada split temporal 
- Setiap variasi dilatih menggunakan konfigurasi RF dan XGB di atas dan dievaluasi pada split temporal 

