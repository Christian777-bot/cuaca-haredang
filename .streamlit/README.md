# Smart Weather Classifier

Website Streamlit untuk memprediksi tipe cuaca menggunakan model Machine Learning/XGBoost.

## Isi Website

- Prediksi tipe cuaca: Berawan, Hujan, Bersalju, atau Cerah
- Tingkat keyakinan model dalam persen
- Rekomendasi praktis untuk pengguna awam
- Grafik peluang setiap kelas cuaca
- Penjelasan sederhana kenapa model mengarah ke prediksi tersebut
- Evaluasi model: Accuracy, F1-Macro, Precision, Recall

## Struktur Folder

```text
app.py
requirements.txt
runtime.txt
weather_classification_data.csv
models/
  weather_xgboost.joblib
  metrics.json
src/
  train_model.py
.streamlit/
  config.toml
```

## Cara Deploy di Streamlit Cloud

1. Upload semua file dan folder ke GitHub.
2. Pastikan file model berada di `models/weather_xgboost.joblib`.
3. Di Streamlit Cloud, pilih repository ini.
4. Main file path: `app.py`.
5. Klik Deploy atau Reboot app.

## Catatan

Hasil prediksi berasal dari model Machine Learning berdasarkan dataset, bukan data cuaca resmi real-time.
