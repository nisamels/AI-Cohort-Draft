# SmartBudgeting Robo-Advisor

**SmartBudgeting Robo-Advisor** adalah sistem rekomendasi investasi berbasis *deep learning* yang dirancang untuk memberikan saran alokasi aset (Reksadana, Obligasi, Saham) secara personal berdasarkan profil keuangan pengguna. Model dikembangkan menggunakan TensorFlow 2.x dengan pendekatan *multi‑task learning* (regresi + klasifikasi + alokasi aset) serta komponen kustom untuk menangani ketidakseimbangan data dan fokus pada segmen berisiko tinggi.

---

##  Ringkasan Proyek

Proyek ini merupakan implementasi dari *main quest* yang mewajibkan:

1. **Custom Layer** – `FinancialAttentionLayer` (feature‑wise attention dengan parameter trainable).
2. **Custom Loss Function** – `WeightedHuberLoss` (Huber loss dengan dukungan sample weight).
3. **Custom Callback** – `SegmentMAECallback` (monitor MAE per segmen saving ratio setiap epoch).
4. **Model Deep Learning** – menggunakan TensorFlow Functional API dengan tiga output:
   - `ratio_output` : prediksi expense_ratio (regresi, sigmoid)
   - `cat_output`   : prediksi budget status (klasifikasi 3 kelas, softmax)
   - `alloc_output` : prediksi alokasi aset (3 kelas softmax → persentase Reksadana, Obligasi, Saham)
5. **Inference pipeline sederhana** – fungsi `predict_allocation()` yang menerima input 7 fitur mentah dan mengembalikan rekomendasi alokasi aset beserta saving ratio dan profil risiko.

Dataset yang digunakan berasal dari `databersihowi.csv` (31.856 sampel) dengan fitur utama: income tahunan, total expense bulanan, lifestyle expense, loan interest rate, credit score, family size, dan usia.

---

## Perbandingan Versi v2 vs Versi Final (smartbudgetfinal)

| Aspek | Versi v2 (`SmartBudgeting_RoboAdvisor_v2`) | Versi Final (`smartbudgetfinal`) | Alasan Perbaikan (Saran Advisor) |
|-------|---------------------------------------------|----------------------------------|----------------------------------|
| **Target Regresi** | `saving_ratio` (sangat skewed, 93,8% > 50%) | `expense_ratio` (distribusi lebih merata) | Menghindari bias prediksi ke nilai tinggi. |
| **Jumlah Output Model** | 2 (ratio + klasifikasi) | 3 (ratio + klasifikasi + **alokasi aset**) | Advisor meminta alokasi aset juga dihasilkan oleh ML, bukan rule‑based. |
| **Custom Loss** | `HybridFinancialLoss` (Huber + MAE + penalty) | `WeightedHuberLoss` (Huber + sample weight) | Fokus pada error yang diberi bobot (data minoritas expense_ratio tinggi). |
| **Sample Weight** | Hanya untuk klasifikasi | Bobot regresi ekstrim (inverse frequency kuadrat) × class weight | Memperkuat sinyal dari data expense_ratio tinggi (hanya ~7% data). |
| **Arsitektur Model** | 2 head: regresi (64→32→1), klasifikasi (32→3) | 3 head: regresi (32→1), klasifikasi (32→3), alokasi (64→32→3 softmax) | Kapasitas lebih besar untuk multitask. |
| **Callback** | `FinancialHealthMonitor` (monitor distribusi prediksi) | `SegmentMAECallback` (monitor MAE per segmen saving ratio) | Lebih informatif untuk mengejar MAE rendah. |
| **Loss Weight** | ratio : 15, cat : 1 | ratio : 30, cat : 1, alloc : 10 | Regresi dan alokasi diberi bobot lebih besar. |
| **Inference Pipeline** | `predict_profile_v2` (alokasi masih pakai rule‑based `get_allocation`) | `predict_allocation` (alokasi murni dari `alloc_output` model) | **Menghilangkan if‑else** → sesuai saran advisor. |
| **Akurasi Klasifikasi** | 85% | 77% (sementara, karena fokus ke alokasi) | Masih perlu tuning, namun arsitektur sudah benar. |
| **Alokasi Aset** | Ditentukan oleh aturan bisnis (if‑else) | Diprediksi langsung oleh model (softmax) | **End‑to‑end ML** seperti yang diminta. |

> **Catatan:** Versi final berhasil memenuhi seluruh persyaratan teknis, meskipun performa awal (akurasi klasifikasi 77%) masih di bawah target 85% karena prioritas diberikan pada keberhasilan arsitektur dan penghilangan rule‑based. Peningkatan performa dapat dilakukan dengan oversampling data minoritas atau tuning lebih lanjut.

---

##  Pemenuhan Main Quest

| Persyaratan | Implementasi | Status |
|-------------|--------------|--------|
| **Custom Layer** | `FinancialAttentionLayer` (units=128, temperature=1.5, dropout=0.2) | ✔️ |
| **Custom Loss Function** | `WeightedHuberLoss` (delta=0.05, mendukung sample_weight) | ✔️ |
| **Custom Callback** | `SegmentMAECallback` (monitor MAE per segmen setiap 5 epoch) | ✔️ |
| **Model Deep Learning (TF Functional API)** | `build_model_allocation_focus()` → `keras.Input` + `Model` dengan 3 output | ✔️ |
| **Inference Pipeline Sederhana** | `predict_allocation()` (feature engineering → scaling → prediksi → parsing hasil) | ✔️ |
| **Output Alokasi Aset** | Proporsi Reksadana, Obligasi, Saham (total 100%) dari `alloc_output` softmax | ✔️ |

---

## 🚀 Cara Menjalankan

1. Clone repository ini dan pastikan memiliki file `databersihowi.csv` di direktori yang sama.
2. Jalankan notebook `smartbudgetfinal.ipynb` secara berurutan.
3. Proses akan:
   - Load data, feature engineering, split, scaling.
   - Membangun model dengan komponen kustom.
   - Melatih model (epoch dapat disesuaikan).
   - Mengevaluasi dengan test set (MAE, akurasi, confusion matrix, MAE per segment).
   - Menyimpan model (`smart_budgeting_end2end.keras`) dan artifacts (`scaler_end2end.pkl`, `features_end2end.pkl`).
   - Menjalankan batch inference untuk contoh user.

Contoh penggunaan inference:

```python
sample_user = {
    'income': 120000,
    'total_expense': 350,
    'lifestyle_expense': 300,
    'loan_int_rate': 8.5,
    'credit_score': 780,
    'family_size': 1,
    'age': 28
}
result = predict_allocation(sample_user, model, scaler)
print(f"Profil: {result['profil']}")
print(f"Alokasi: Saham {result['asset_allocation']['Saham']}%, Obligasi {result['asset_allocation']['Obligasi']}%, Reksadana {result['asset_allocation']['Reksadana']}%")
