# Breakdown Riset Detail dan MVP Penelitian

## 1. Arah Final Penelitian

### Topik final yang disarankan

Comparative Analysis of Machine Learning Models for Salary Range Classification in HR Analytics

### Rumusan sederhana penelitian

Penelitian ini bertujuan membangun model multiclass classification untuk memprediksi kategori salary range berdasarkan atribut pekerjaan dan karakteristik profesional, lalu membandingkan performa beberapa model machine learning untuk menentukan model yang paling efektif dan paling relevan bagi kebutuhan HR analytics.

## 2. Rumusan Masalah

Rumusan masalah yang disarankan:

1. Bagaimana mengubah data salary kontinu menjadi kategori salary range yang layak untuk pemodelan klasifikasi?
2. Bagaimana performa beberapa model machine learning dalam memprediksi kelas salary range?
3. Model mana yang memberikan kinerja terbaik untuk data salary tabular?
4. Fitur apa yang paling berpengaruh terhadap hasil klasifikasi salary range?

## 3. Tujuan Penelitian

1. Menyusun skema kategorisasi salary range berbasis pendekatan statistik.
2. Membangun model klasifikasi salary range menggunakan beberapa algoritma machine learning.
3. Membandingkan performa model menggunakan metrik evaluasi multiclass.
4. Menganalisis fitur dominan yang memengaruhi prediksi kelas salary.

## 4. Hipotesis Kerja

Hipotesis operasional yang bisa dipakai:

1. model boosting berbasis tree akan memberikan performa lebih baik daripada baseline linear pada data salary tabular,
2. kategorisasi salary berbasis quartile akan menghasilkan distribusi kelas yang lebih seimbang,
3. fitur seperti pengalaman kerja, job title, lokasi, dan tingkat pendidikan memiliki pengaruh besar terhadap kelas salary.

## 5. Variabel Penelitian

### 5.1 Variabel target

Target penelitian adalah salary range class.

Contoh pembentukan kelas:

1. Q1: Low
2. Q2: Medium
3. Q3: High
4. Q4: Premium

### 5.2 Variabel fitur

Fitur menyesuaikan dataset, tetapi minimal dikelompokkan menjadi:

1. fitur demografis atau profesional,
2. fitur pendidikan,
3. fitur pengalaman kerja,
4. fitur pekerjaan,
5. fitur perusahaan atau lokasi,
6. fitur kompetensi atau skill bila tersedia.

Contoh fitur yang layak:

1. years_experience,
2. education_level,
3. employment_type,
4. company_location,
5. employee_residence,
6. company_size,
7. remote_ratio,
8. job_title,
9. industry,
10. skills_count.

## 6. Dataset yang Disarankan

Supaya cepat dan aman untuk Sinta 3, gunakan dataset publik yang umum dipakai dan mudah direplikasi. Opsi paling realistis:

1. Data Science Job Salaries dataset,
2. Tech salary dataset dengan fitur pekerjaan dan lokasi,
3. salary survey dataset dengan atribut pekerjaan dan pendidikan.

Kriteria dataset:

1. memiliki kolom salary atau annual salary,
2. memiliki fitur campuran numerik dan kategorikal,
3. jumlah data minimal cukup untuk multiclass,
4. sumbernya dapat disitasi.

Jika Anda belum menetapkan dataset, pilih satu dataset publik yang bersih dan cukup populer. Jangan gunakan terlalu banyak dataset untuk target Sinta 3 kecuali memang sudah tersedia dan siap diproses.

## 7. Breakdown Tahapan Penelitian

## Tahap 1. Studi Literatur

### Tujuan

Menyusun latar belakang, gap penelitian, dan dasar pemilihan metode.

### Fokus bacaan

1. salary prediction,
2. salary classification,
3. HR analytics,
4. multiclass classification,
5. explainable AI pada data tabular.

### Output

1. matriks penelitian terdahulu,
2. research gap,
3. alasan pemilihan model dan metrik.

## Tahap 2. Pemahaman Data

### Aktivitas

1. identifikasi struktur dataset,
2. cek missing values,
3. cek duplikasi,
4. cek distribusi target salary,
5. identifikasi cardinality fitur kategorikal,
6. cek ketimpangan distribusi kandidat kelas.

### Output

1. deskripsi dataset,
2. tabel karakteristik fitur,
3. visualisasi awal.

## Tahap 3. Pembentukan Target Kelas

### Aktivitas

1. pilih metode discretization salary,
2. gunakan quartile sebagai default,
3. buat label Low, Medium, High, Premium,
4. cek kembali distribusi kelas.

### Catatan

Untuk Sinta 3, quartile adalah pilihan paling aman karena mudah dijustifikasi secara statistik dan cenderung menjaga distribusi kelas tetap lebih seimbang.

### Output

1. tabel batas kelas salary,
2. visualisasi distribusi kelas,
3. argumentasi pemilihan discretization.

## Tahap 4. Preprocessing

### Aktivitas

1. imputasi missing value,
2. penghapusan duplikasi,
3. handling outlier bila perlu,
4. encoding fitur kategorikal,
5. scaling untuk model yang memerlukan,
6. train-test split secara stratified.

### Strategi encoding

1. one-hot encoding untuk fitur kategorikal kecil,
2. frequency encoding untuk cardinality tinggi,
3. gunakan native categorical handling jika memakai CatBoost.

### Output

1. dataset siap modeling,
2. pipeline preprocessing yang dapat direplikasi.

## Tahap 5. Baseline Modeling

### Tujuan

Menyediakan titik acuan yang sederhana.

### Model baseline

1. Logistic Regression,
2. Decision Tree.

### Output

1. skor baseline,
2. bukti bahwa model lanjutan memang memberi peningkatan.

## Tahap 6. Advanced Modeling

### Model utama yang disarankan

1. Random Forest atau XGBoost,
2. CatBoost,
3. ANN opsional.

### Rekomendasi praktis

Jika ingin jalur paling efisien:

1. Logistic Regression,
2. Random Forest,
3. CatBoost.

Jika ingin sedikit lebih kuat:

1. Logistic Regression,
2. XGBoost,
3. CatBoost.

## Tahap 7. Evaluasi Model

### Metrik utama

1. macro F1-score,
2. weighted F1-score,
3. accuracy,
4. confusion matrix.

### Metrik tambahan

1. multiclass ROC-AUC,
2. precision dan recall per kelas,
3. stratified cross validation.

### Alasan

Macro F1 harus menjadi metrik utama karena lebih adil untuk multiclass yang mungkin tidak benar-benar seimbang.

## Tahap 8. Interpretasi Model

### Tujuan

Menjelaskan kenapa model memprediksi kelas salary tertentu.

### Pilihan metode

1. feature importance bawaan model,
2. permutation importance,
3. SHAP untuk model terbaik.

### Output

1. daftar fitur dominan,
2. pembahasan relevansi fitur untuk HR analytics.

## Tahap 9. Pembahasan

Poin pembahasan yang harus muncul:

1. kenapa model tertentu unggul,
2. bagaimana karakter fitur memengaruhi hasil,
3. kelas mana yang paling sulit diprediksi,
4. implikasi hasil untuk salary benchmarking,
5. keterbatasan dataset dan generalisasi.

## 8. MVP Penelitian

## 8.1 Definisi MVP

MVP dalam konteks penelitian ini adalah versi eksperimen minimum yang sudah cukup layak untuk menghasilkan artikel Sinta 3 yang rapi, walaupun belum terlalu kompleks.

## 8.2 Komponen MVP yang wajib ada

1. satu dataset publik yang jelas,
2. satu skema salary range berbasis quartile,
3. tiga model klasifikasi,
4. satu baseline sederhana,
5. metrik accuracy, macro F1, weighted F1, confusion matrix,
6. satu analisis feature importance pada model terbaik,
7. narasi pembahasan domain HR.

## 8.3 Stack MVP yang direkomendasikan

### Dataset

Satu dataset salary publik.

### Target kelas

4 kelas berbasis quartile.

### Model

1. Logistic Regression,
2. Random Forest atau XGBoost,
3. CatBoost.

### Evaluasi

1. train-test split stratified,
2. macro F1 sebagai metrik utama,
3. confusion matrix,
4. feature importance.

### Deliverable eksperimen

1. tabel distribusi kelas,
2. tabel hasil perbandingan model,
3. confusion matrix model terbaik,
4. grafik feature importance,
5. draft pembahasan.

## 8.4 Apa yang tidak wajib pada MVP

Komponen berikut bisa ditunda jika waktu terbatas:

1. ANN,
2. Optuna,
3. SHAP untuk semua model,
4. SMOTE yang kompleks,
5. eksperimen multi-dataset,
6. ensemble stacking.

## 9. Rencana Eksekusi Praktis

## Minggu 1

1. finalisasi judul,
2. finalisasi dataset,
3. susun literatur inti,
4. buat kategorisasi salary range.

## Minggu 2

1. preprocessing,
2. EDA,
3. baseline modeling,
4. training model utama.

## Minggu 3

1. evaluasi hasil,
2. analisis feature importance,
3. susun tabel dan gambar hasil,
4. tulis pembahasan.

## Minggu 4

1. rapikan draft artikel,
2. sinkronkan sitasi,
3. pindahkan ke template jurnal,
4. lakukan proofreading akhir.

## 10. Struktur Output Penelitian

Output minimal yang perlu tersedia:

1. file dataset bersih atau script preprocessing,
2. notebook atau script training model,
3. tabel hasil eksperimen,
4. grafik evaluasi,
5. draft artikel.

## 11. Checklist Siap Tulis

Checklist yang harus tercentang sebelum dipindah ke template:

1. istilah multiclass sudah konsisten,
2. judul final sudah dipilih,
3. dataset dan sumber sitasi sudah final,
4. batas kelas salary sudah final,
5. model eksperimen final sudah ditetapkan,
6. tabel hasil sudah tersedia,
7. pembahasan fitur dominan sudah ditulis,
8. abstrak sudah sesuai hasil,
9. kesimpulan tidak melebihi hasil eksperimen.

## 12. Formula Final yang Direkomendasikan

Jika Anda ingin jalur paling efisien untuk cepat menjadi draft yang kuat, pakai formula ini:

1. dataset salary publik tunggal,
2. salary dibagi 4 kelas dengan quartile,
3. model: Logistic Regression, XGBoost atau Random Forest, CatBoost,
4. metrik utama: macro F1,
5. interpretasi: feature importance atau SHAP pada model terbaik,
6. kontribusi: comparative multiclass salary classification untuk HR analytics.

Itu adalah MVP yang cukup kuat untuk target Sinta 3 dan realistis untuk dieksekusi end-to-end.