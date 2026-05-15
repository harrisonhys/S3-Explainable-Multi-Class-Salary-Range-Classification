# Dataset Plan Penelitian Salary Range Classification

## 1. Tujuan Dataset Plan

Dokumen ini disusun untuk memastikan pemilihan dataset tidak mengganggu arah artikel. Untuk target Sinta 3, dataset harus memenuhi tiga syarat utama:

1. relevan langsung dengan salary prediction atau salary range classification,
2. memiliki fitur tabular campuran numerik dan kategorikal,
3. mudah direplikasi dan mudah dijelaskan dalam metodologi.

## 2. Kriteria Dataset yang Dibutuhkan

Dataset yang layak dipakai sebaiknya memenuhi kriteria berikut:

1. memiliki variabel salary numerik yang dapat diubah menjadi kelas,
2. memiliki jumlah sampel cukup untuk empat kelas salary,
3. memiliki fitur pekerjaan yang realistis untuk domain HR analytics,
4. sumber dataset dapat disitasi dengan jelas,
5. preprocessing tidak terlalu ekstrem agar penelitian tetap fokus pada kontribusi utama.

## 3. Kandidat Dataset

## Opsi A. Data Science Job Salaries Dataset

### Deskripsi singkat

Dataset ini umum dipakai pada analisis salary data science dan biasanya tersedia di Kaggle dengan nama serupa seperti ds_salaries atau data science job salaries. Kolom yang sering tersedia antara lain:

1. work_year,
2. experience_level,
3. employment_type,
4. job_title,
5. salary,
6. salary_currency,
7. salary_in_usd,
8. employee_residence,
9. remote_ratio,
10. company_location,
11. company_size.

### Kelebihan

1. sangat relevan dengan salary prediction,
2. punya target salary numerik yang bisa langsung didiscretize,
3. fitur kategorikalnya kuat untuk CatBoost,
4. mudah dijelaskan dalam konteks HR analytics,
5. cocok untuk studi komparatif multiclass classification.

### Kekurangan

1. domainnya spesifik ke data science jobs,
2. kemungkinan ada ketimpangan distribusi pada job title tertentu,
3. perlu perhatian pada salary_currency dan salary_in_usd agar tidak redundan.

### Kelayakan untuk penelitian ini

Sangat layak dan menjadi kandidat utama.

## Opsi B. Adult Income Dataset

### Deskripsi singkat

Dataset Adult Income dari UCI sangat populer untuk klasifikasi pendapatan. Fitur umumnya meliputi:

1. age,
2. workclass,
3. education,
4. marital-status,
5. occupation,
6. relationship,
7. race,
8. sex,
9. capital-gain,
10. capital-loss,
11. hours-per-week,
12. native-country,
13. income class.

### Kelebihan

1. sangat populer dan mudah direplikasi,
2. banyak referensi pembanding,
3. preprocessing relatif standar,
4. cocok untuk baseline eksperimen cepat.

### Kekurangan

1. target sudah berupa klasifikasi biner income, bukan salary numerik kontinu,
2. tidak benar-benar mendukung skenario salary range classification empat kelas,
3. lebih cocok untuk income classification daripada salary range modeling.

### Kelayakan untuk penelitian ini

Layak sebagai referensi metodologis atau baseline tambahan, tetapi kurang ideal sebagai dataset utama.

## Opsi C. Salary Survey atau Developer Compensation Dataset

### Deskripsi singkat

Kelompok dataset ini biasanya berasal dari salary survey atau developer compensation survey dan berisi informasi seperti:

1. country,
2. education,
3. years of coding,
4. years of professional experience,
5. developer role,
6. industry,
7. annual compensation.

### Kelebihan

1. dekat dengan konteks pasar kerja nyata,
2. kaya fitur profesional,
3. cocok untuk HR analytics dan salary benchmarking.

### Kekurangan

1. sering membutuhkan pembersihan besar,
2. banyak missing values,
3. definisi kolom sering tidak konsisten,
4. waktu preprocessing bisa membesar dan mengganggu jadwal penelitian.

### Kelayakan untuk penelitian ini

Layak hanya jika Anda memang sudah memiliki dataset yang bersih. Jika belum, opsi ini tidak efisien untuk target Sinta 3.

## 4. Rekomendasi Dataset Final

### Dataset yang direkomendasikan

**Data Science Job Salaries Dataset**

### Alasan pemilihan

1. target salary numerik tersedia dan mudah diubah ke empat kelas salary range,
2. fitur sangat sesuai dengan konteks penelitian,
3. preprocessing masih masuk akal,
4. hasil mudah dibahas dalam perspektif HR analytics,
5. model seperti CatBoost, XGBoost, dan Logistic Regression bisa dibandingkan secara adil.

## 5. Rencana Penggunaan Kolom

## 5.1 Target yang digunakan

Gunakan kolom salary_in_usd sebagai target utama karena:

1. sudah distandarkan ke USD,
2. menghindari noise dari banyak mata uang,
3. lebih konsisten untuk pembentukan kelas.

### Strategi pembentukan target

Transformasikan salary_in_usd menjadi empat kelas berbasis quartile:

1. Low,
2. Medium,
3. High,
4. Premium.

## 5.2 Fitur yang direkomendasikan

Gunakan fitur berikut sebagai kandidat utama:

1. work_year,
2. experience_level,
3. employment_type,
4. job_title,
5. employee_residence,
6. remote_ratio,
7. company_location,
8. company_size.

## 5.3 Fitur yang sebaiknya tidak langsung dipakai

1. salary,
2. salary_currency,
3. kolom lain yang merupakan turunan langsung dari target.

Alasannya untuk mencegah leakage.

## 6. Rencana Preprocessing Dataset

Tahap preprocessing yang direkomendasikan:

1. cek ukuran data dan tipe fitur,
2. hapus duplikasi,
3. cek missing values,
4. gunakan salary_in_usd sebagai target numerik mentah,
5. bentuk empat kelas salary dengan quartile,
6. pisahkan fitur numerik dan kategorikal,
7. gunakan one-hot encoding untuk Logistic Regression,
8. gunakan native categorical handling untuk CatBoost,
9. lakukan stratified train-test split,
10. evaluasi distribusi kelas hasil discretization.

## 7. Rencana Eksperimen Dataset

## 7.1 Versi MVP

Versi MVP eksperimen dataset:

1. satu dataset utama,
2. satu skema discretization quartile,
3. tiga model klasifikasi,
4. satu tabel hasil perbandingan,
5. satu analisis fitur dominan.

## 7.2 Versi penguatan jika waktu cukup

1. bandingkan quartile dengan percentile tertentu,
2. tambah cross validation,
3. tambahkan SHAP untuk model terbaik,
4. tambahkan analisis confusion per kelas.

## 8. Potensi Masalah dan Mitigasi

## Masalah 1. Job title terlalu banyak kategori

Mitigasi:

1. grouping kategori yang sangat jarang,
2. frequency encoding untuk model tertentu,
3. prioritaskan CatBoost untuk eksperimen utama.

## Masalah 2. Distribusi salary skewed

Mitigasi:

1. pakai quartile,
2. tampilkan histogram salary,
3. tampilkan distribusi kelas setelah discretization.

## Masalah 3. Leakage dari fitur salary terkait

Mitigasi:

1. jangan gunakan salary mentah dan salary_currency sebagai fitur,
2. validasi ulang seluruh kolom sebelum training.

## Masalah 4. Generalisasi terbatas karena domain data science

Mitigasi:

1. nyatakan ini secara eksplisit di bagian keterbatasan,
2. bingkai kontribusi sebagai studi terapan pada domain pekerjaan data science,
3. sarankan pengujian lintas domain pada penelitian berikutnya.

## 9. Narasi Siap Pakai untuk Metodologi

Narasi yang bisa dipindahkan ke draft:

Penelitian ini menggunakan Data Science Job Salaries Dataset sebagai sumber data utama karena menyediakan variabel salary terstandarisasi dan atribut profesional yang relevan untuk analisis kompensasi. Variabel target yang digunakan adalah salary_in_usd, yang kemudian ditransformasikan menjadi empat kelas salary range menggunakan pendekatan quartile-based discretization. Fitur prediktor meliputi karakteristik pekerjaan, pengalaman, jenis pekerjaan, lokasi karyawan, tingkat remote work, lokasi perusahaan, dan ukuran perusahaan. Pemilihan dataset ini dipertimbangkan karena relevan langsung dengan salary analytics, mudah direplikasi, dan memiliki kombinasi fitur numerik-kategorikal yang sesuai untuk evaluasi model machine learning tabular.

## 10. Keputusan Final yang Disarankan

Untuk menjaga penelitian tetap fokus, dataset plan final yang direkomendasikan adalah:

1. dataset utama: Data Science Job Salaries Dataset,
2. target: salary_in_usd,
3. formulasi target: multiclass salary range classification dengan 4 kelas quartile,
4. model utama: Logistic Regression, XGBoost atau Random Forest, dan CatBoost,
5. fokus pembahasan: performa model dan interpretasi fitur dalam konteks HR analytics.

Dengan keputusan ini, draft artikel akan lebih siap masuk ke Bab Pendahuluan, Metodologi, dan Hasil.