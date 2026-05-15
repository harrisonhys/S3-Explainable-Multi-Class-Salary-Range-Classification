# Analisis Penelitian Salary Classification untuk Target Sinta 3

## 1. Ringkasan Posisi Penelitian

Ide yang sudah ditulis memiliki inti yang layak untuk target Sinta 3, yaitu membangun model klasifikasi rentang gaji berbasis data tabular dan membandingkan beberapa algoritma machine learning yang umum dipakai pada data terstruktur. Secara substansi, topik ini lebih kuat jika diposisikan sebagai penelitian terapan pada domain HR analytics daripada penelitian novelty algoritmik.

Posisi yang paling aman untuk Sinta 3 adalah:

1. fokus pada masalah praktis yang relevan,
2. dataset jelas dan dapat direplikasi,
3. eksperimen rapi dan cukup lengkap,
4. pembahasan tidak hanya berhenti di akurasi,
5. ada interpretasi faktor yang memengaruhi kelas salary.

## 2. Koreksi Framing yang Penting

Ada satu isu konseptual yang perlu dibetulkan sejak awal:

- folder dan nama proyek memakai istilah multilabel,
- tetapi ide penelitian Anda bukan multilabel classification,
- penelitian ini adalah multiclass classification.

Penjelasannya sederhana:

- multiclass: satu data hanya masuk ke satu kelas salary, misalnya LOW atau MEDIUM atau HIGH,
- multilabel: satu data bisa punya beberapa label sekaligus.

Karena target output Anda adalah satu kelas salary range untuk tiap sampel, maka istilah yang benar adalah multiclass salary classification atau salary range classification.

Jika istilah ini tidak dikoreksi, reviewer bisa menganggap fondasi metodologinya kurang rapi.

## 3. Kelayakan untuk Sinta 3

Topik ini layak untuk Sinta 3 karena memenuhi karakter penelitian terapan yang umum diterima, yaitu:

1. masalah nyata dan mudah dipahami,
2. penggunaan metode machine learning yang relevan untuk data tabular,
3. ada perbandingan model,
4. ada proses evaluasi terukur,
5. ada peluang interpretasi hasil untuk pengambilan keputusan HR.

Namun, untuk Sinta 3, penelitian ini akan lebih kuat jika tidak terlalu memaksakan klaim novelty tinggi. Yang lebih aman adalah menonjolkan:

1. perubahan pendekatan dari regression ke salary range classification,
2. perbandingan model yang sesuai dengan karakter data tabular campuran numerik-kategorikal,
3. analisis explainability untuk mendukung keputusan HR,
4. rekomendasi model praktis terbaik berdasarkan performa dan interpretabilitas.

## 4. Nilai Kuat dari Ide Saat Ini

Beberapa komponen ide Anda sudah bagus dan perlu dipertahankan:

1. masalah penelitian cukup aplikatif untuk HR analytics,
2. framing salary range lebih realistis dibanding exact salary regression,
3. pemilihan CatBoost dan XGBoost masuk akal untuk data tabular,
4. evaluasi macro F1 dan confusion matrix sudah tepat untuk multiclass,
5. komponen feature importance atau SHAP memberi nilai tambah pembahasan.

## 5. Bagian yang Perlu Disederhanakan untuk Sinta 3

Supaya lebih realistis dan cepat menjadi artikel siap submit, beberapa bagian perlu disederhanakan.

### 5.1 Jangan terlalu banyak klaim research gap

Gap yang aman untuk dipakai:

1. mayoritas studi salary prediction fokus pada regresi,
2. studi klasifikasi rentang gaji masih lebih terbatas,
3. belum banyak pembahasan model yang praktis dan interpretatif untuk HR decision support pada skenario salary range.

Gap yang sebaiknya tidak diklaim terlalu keras tanpa literature review kuat:

1. belum ada evaluasi komprehensif XGBoost, CatBoost, dan ANN,
2. benar-benar belum ada analisis fitur untuk salary range classification.

Klaim seperti itu harus dibuktikan oleh telaah pustaka yang sangat sistematis. Untuk Sinta 3, Anda tidak perlu mengambil risiko sebesar itu.

### 5.2 ANN bukan komponen wajib

ANN bisa dipakai, tetapi untuk target Sinta 3, model utama yang paling rasional adalah:

1. Logistic Regression atau Decision Tree sebagai baseline,
2. Random Forest atau XGBoost,
3. CatBoost.

ANN boleh menjadi model tambahan, bukan pusat kontribusi. Pada data tabular ukuran kecil hingga menengah, ANN sering tidak unggul dan justru menambah beban tuning.

### 5.3 Explainability cukup satu metode utama

Tidak perlu memakai terlalu banyak teknik interpretasi sekaligus. Untuk Sinta 3, cukup pilih salah satu:

1. SHAP,
2. permutation importance,
3. built-in feature importance.

Pilihan paling kuat secara akademik tetap SHAP, tetapi jika waktu eksekusi terbatas, gunakan feature importance bawaan model plus pembahasan yang rapi.

## 6. Posisi Kontribusi yang Disarankan

Kontribusi penelitian sebaiknya ditulis secara sederhana dan defensible:

1. mengusulkan formulasi salary prediction sebagai masalah multiclass salary range classification,
2. menyajikan perbandingan beberapa model machine learning pada data salary tabular,
3. mengidentifikasi fitur paling berpengaruh terhadap kelas salary,
4. memberikan rekomendasi model yang paling sesuai untuk kebutuhan HR analytics praktis.

Kontribusi ini cukup untuk Sinta 3 dan tidak terdengar berlebihan.

## 7. Struktur Artikel yang Paling Aman untuk Sinta 3

Susunan artikel yang disarankan:

1. Pendahuluan
2. Tinjauan pustaka
3. Metodologi penelitian
4. Hasil dan pembahasan
5. Kesimpulan

Isi yang harus kuat:

1. alasan mengubah salary kontinu menjadi kelas,
2. alasan pemilihan batas kelas salary,
3. alasan pemilihan model,
4. alasan pemilihan metrik multiclass,
5. interpretasi hasil terhadap domain HR.

## 8. Judul yang Lebih Aman untuk Sinta 3

Judul yang direkomendasikan:

### Opsi utama

Comparative Analysis of Machine Learning Models for Salary Range Classification in HR Analytics

### Opsi alternatif 1

Multi-Class Salary Range Classification Using XGBoost, CatBoost, and Baseline Machine Learning Models

### Opsi alternatif 2

Explainable Salary Range Classification for HR Decision Support Using Machine Learning

Untuk target Sinta 3, judul pertama paling aman karena:

1. tidak berlebihan,
2. jelas menunjukkan comparative study,
3. mudah dipahami reviewer,
4. sesuai dengan kontribusi terapan.

## 9. Desain Eksperimen yang Disarankan

Desain eksperimen minimum yang cukup kuat:

1. dataset salary publik yang jelas sumbernya,
2. pembersihan data dan transformasi target ke empat kelas,
3. baseline model: Logistic Regression,
4. model pembanding utama: Random Forest atau XGBoost,
5. model utama terbaik untuk data kategorikal: CatBoost,
6. evaluasi dengan accuracy, macro F1, weighted F1, confusion matrix, dan multiclass ROC-AUC jika memungkinkan,
7. interpretasi fitur penting dari model terbaik.

Jika sumber daya waktu cukup, tambahkan:

1. Stratified K-Fold cross validation,
2. hyperparameter tuning ringan,
3. SHAP pada model terbaik.

## 10. Risiko Reviewer dan Cara Mengamankannya

### Risiko 1: novelty dianggap rendah

Mitigasi:

1. tekankan kontribusi terapan,
2. tampilkan perbandingan model yang rapi,
3. sertakan analisis faktor dominan,
4. hubungkan hasil ke konteks HR decision support.

### Risiko 2: kelas salary dibuat terlalu arbitrer

Mitigasi:

1. gunakan quartile atau percentile,
2. jelaskan alasan statistiknya,
3. tampilkan distribusi tiap kelas.

### Risiko 3: data imbalance membuat hasil bias

Mitigasi:

1. gunakan stratified split,
2. prioritaskan macro F1,
3. tampilkan confusion matrix,
4. jika perlu gunakan class weighting atau SMOTE hanya pada data training.

### Risiko 4: terlalu banyak model tapi pembahasan tipis

Mitigasi:

1. batasi model inti menjadi 3 sampai 4,
2. fokus pada alasan kenapa model tertentu unggul,
3. diskusikan trade-off performa dan interpretabilitas.

## 11. Rekomendasi Final

Versi penelitian yang paling aman untuk segera dibawa ke draft Sinta 3 adalah:

1. ubah framing menjadi multiclass salary range classification,
2. gunakan 4 kelas salary berbasis quartile,
3. bandingkan 3 sampai 4 model saja,
4. prioritaskan CatBoost, XGBoost atau Random Forest, dan satu baseline sederhana,
5. gunakan macro F1 sebagai metrik utama,
6. tambahkan interpretasi fitur,
7. tulis pembahasan yang kuat pada aspek HR analytics.

## 12. Keputusan Editorial yang Disarankan

Jika tujuan Anda adalah cepat menjadi artikel siap template, maka versi terbaik bukan penelitian yang paling kompleks, melainkan penelitian yang paling rapi, konsisten, dan selesai end-to-end.

Karena itu, formulasi yang direkomendasikan adalah:

**penelitian komparatif multiclass salary classification berbasis machine learning tabular dengan fokus pada performa dan interpretabilitas untuk HR decision support.**

Itu cukup kuat untuk target Sinta 3, realistis untuk dikerjakan, dan aman secara akademik.