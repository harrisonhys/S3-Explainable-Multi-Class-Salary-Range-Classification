# Matriks Penelitian Terdahulu

## 1. Tujuan Dokumen

Dokumen ini disusun untuk membantu penulisan Bab Pendahuluan dan Tinjauan Pustaka. Fokusnya bukan sekadar mengumpulkan paper, tetapi memetakan posisi penelitian Anda terhadap studi sebelumnya agar research gap lebih rapi dan tidak berlebihan.

## 2. Prinsip Penyusunan Matriks

Matriks ini memadukan tiga kelompok studi:

1. studi salary prediction atau salary benchmarking,
2. studi income classification yang relevan secara metodologis,
3. studi explainable machine learning dan model boosting yang relevan dengan rancangan eksperimen.

Dengan pendekatan ini, Anda tidak perlu memaksa semua penelitian terdahulu harus identik 100% dengan topik Anda. Yang penting adalah hubungan metodologis dan substantifnya jelas.

## 3. Matriks Penelitian Terdahulu

| No | Peneliti dan Tahun | Fokus Penelitian | Formulasi Masalah | Metode Utama | Data / Konteks | Temuan Inti | Relevansi ke Penelitian Ini | Celah terhadap Penelitian Ini |
|---|---|---|---|---|---|---|---|---|
| 1 | Meng et al. (2018) | Intelligent salary benchmarking for talent recruitment | Salary benchmarking dan estimation | Holistic matrix factorization | Data rekrutmen online dan job-company similarity | Menunjukkan salary benchmarking dapat dilakukan dengan pendekatan data-driven berbasis kemiripan pekerjaan dan perusahaan | Relevan untuk justifikasi bahwa salary analytics penting untuk recruitment support | Tidak fokus pada multiclass salary range classification dan tidak membandingkan model tabular umum seperti CatBoost atau XGBoost |
| 2 | Matbouli dan Alghamdi (2022) | Salary prediction across economy-wide activities and occupations | Regression salary prediction | Multiple Linear Regression, Tree-Based Regression, SVR, ANN, Gaussian Process Regression | Saudi labor market salary survey | Model nonlinier mengungguli regresi linear; GPR dan ANN menonjol pada skenario tertentu | Sangat relevan untuk menunjukkan dominasi pendekatan regresi pada studi salary prediction | Belum memformulasikan masalah sebagai salary range classification dan belum menekankan interpretasi multiclass untuk HR decision support |
| 3 | Wang et al. (2022) | Factors influencing starting salary of college graduates | Prediksi dan analisis faktor salary awal lulusan | Machine learning-based factor analysis | Starting salary of college graduates | Menunjukkan atribut tertentu berpengaruh terhadap starting salary | Relevan untuk memperkuat argumen bahwa fitur profesional dan pendidikan memengaruhi salary | Fokus utamanya analisis faktor salary awal, bukan komparasi multiclass salary range classification pada data tabular umum |
| 4 | Kuo et al. (2021) | Graduate salary grading prediction using deep learning | Salary grading atau ordinal salary prediction | Deep learning | Graduate salary grading context | Menunjukkan salary dapat dipetakan ke grading, bukan hanya nilai kontinu | Relevan untuk mendukung ide pengubahan salary ke kelas atau grade | Tidak fokus pada comparative study antara model boosting tabular dan baseline klasik |
| 5 | Asaduzzaman dan Uddin et al. (2024) | Novel salary prediction system using machine learning techniques | Salary classification atau salary prediction system | Machine learning classification approaches | Konteks salary prediction system | Menggunakan pendekatan klasifikasi pada tugas salary prediction | Relevan karena menunjukkan arah klasifikasi mulai digunakan dalam penelitian salary | Perluasan ke multiclass salary range yang terdefinisi statistik dan pembahasan fitur dominan masih terbuka |
| 6 | Kaya, Saatçi, dan Bakal (2024) | Improving salary offer processes with classification-based machine learning models | Klasifikasi untuk salary offer process | Classification-based ML models | Salary offer decision context | Menunjukkan model klasifikasi dapat mendukung proses salary offer | Relevan langsung untuk argumen praktis bahwa klasifikasi lebih aplikatif daripada exact prediction | Belum menonjolkan kerangka multiclass salary range berbasis quartile dan belum diarahkan kuat ke explainability |
| 7 | Mamidala et al. (2024) | Machine learning approaches to salary prediction in HR payroll systems | Income or salary class prediction in HR systems | XGBoost, dibandingkan dengan NB dan SVM | Adult Income Dataset | XGBoost mencapai accuracy 91.16% dan AUC-ROC 0.93, mengungguli model konvensional | Relevan untuk mendukung pemilihan XGBoost sebagai model pembanding kuat | Dataset Adult Income bersifat biner dan bukan salary range numerik empat kelas |
| 8 | Stow (2025) | Explainable machine learning framework for income prediction with class imbalance optimization | Income classification with explainability | XGBoost, LightGBM, Random Forest, CatBoost, SMOTE, explainability | Adult Income Dataset | Menunjukkan explainable ML dan imbalance optimization penting pada prediksi income | Relevan untuk menjustifikasi pentingnya explainability dan evaluasi imbalance | Fokus pada income classification biner, bukan salary range multiclass berbasis data salary nyata |
| 9 | Zita et al. (2025) | Hybrid Bayesian framework for simultaneous job title classification and salary estimation | Joint job classification and salary estimation | Hybrid Bayesian ML | Salary and job title prediction context | Menunjukkan integrasi job classification dan salary estimation dalam satu framework | Relevan untuk menegaskan salary analytics adalah domain riset aktif | Bukan comparative multiclass salary range classification dengan fokus HR analytics praktis |
| 10 | Zeng (2025) | Comparative study of gradient boosting methods for data science salary prediction | Salary prediction comparative study | Gradient boosting methods with CatBoost integration | Data science salary prediction | Menguatkan bahwa CatBoost dan boosting methods kompetitif pada data salary | Relevan untuk memilih CatBoost dan membahas data salary domain data science | Tetap dominan dalam framing prediction umum dan belum diposisikan sebagai multiclass salary range untuk decision support |

## 4. Sintesis Temuan dari Matriks

Berdasarkan matriks di atas, ada beberapa pola penting.

### 4.1 Studi salary prediction masih banyak berfokus pada regresi

Penelitian seperti Matbouli dan Alghamdi menempatkan salary sebagai variabel kontinu dan mengevaluasi model dengan metrik regresi. Ini berguna sebagai landasan, tetapi belum sepenuhnya menjawab kebutuhan HR yang sering bekerja dengan kategori kompensasi.

### 4.2 Pendekatan klasifikasi mulai muncul, tetapi sering masih biner atau kontekstual sempit

Penelitian Mamidala et al. dan Stow menunjukkan klasifikasi income cukup populer, tetapi banyak memakai Adult Income Dataset dengan target biner. Secara metodologis itu relevan, tetapi secara substansi belum sama dengan salary range classification empat kelas.

### 4.3 Studi salary grading atau classification ada, tetapi belum konsisten pada komparasi model tabular praktis

Beberapa paper mulai mengarah ke grading atau classification, misalnya Kuo et al. dan Kaya et al., tetapi belum semuanya menekankan komparasi model yang sederhana, kuat, dan relevan untuk data tabular campuran numerik-kategorikal.

### 4.4 Explainability menjadi peluang penguatan, bukan beban utama

Studi explainable income prediction menunjukkan bahwa interpretasi model memberi nilai tambah. Untuk penelitian Anda, explainability bisa diposisikan sebagai penguatan pembahasan, bukan sebagai pusat novelty.

## 5. Research Gap yang Disarankan

Berikut gap yang paling aman dan defensible untuk dipakai:

1. sebagian besar studi salary prediction masih diformulasikan sebagai masalah regresi,
2. studi klasifikasi yang ada sering berfokus pada income classification biner, bukan salary range multiclass,
3. belum banyak penelitian terapan yang secara spesifik membandingkan model machine learning tabular untuk multiclass salary range classification dalam konteks HR analytics,
4. analisis faktor dominan untuk mendukung interpretasi salary class masih dapat diperkuat.

Gap ini cukup kuat tanpa harus membuat klaim ekstrem seperti “belum pernah ada sama sekali”.

## 6. Posisi Penelitian Anda terhadap Studi Terdahulu

Penelitian yang Anda susun dapat diposisikan sebagai berikut:

1. berbeda dari studi regresi karena target diubah menjadi kelas salary range,
2. berbeda dari income classification biner karena target dibuat multiclass empat kategori,
3. berbeda dari studi deep learning murni karena fokus pada model tabular yang lebih realistis untuk dataset campuran,
4. berbeda dari studi yang hanya mengejar akurasi karena juga membahas faktor yang memengaruhi kelas salary.

## 7. Narasi Siap Pakai untuk Bab Pendahuluan

Narasi yang dapat dipakai atau disesuaikan:

Sebagian besar penelitian salary prediction masih memformulasikan masalah sebagai tugas regresi untuk memperkirakan nilai salary secara kontinu. Pendekatan tersebut penting untuk estimasi numerik, tetapi tidak selalu sejalan dengan kebutuhan praktis HR analytics yang lebih sering bekerja dengan kategori kompensasi untuk keperluan benchmarking, screening, dan decision support. Di sisi lain, studi klasifikasi pada domain income memang telah banyak dilakukan, tetapi mayoritas masih berfokus pada klasifikasi biner seperti level income tinggi dan rendah. Dengan demikian, masih terdapat ruang penelitian untuk mengembangkan pendekatan multiclass salary range classification yang lebih aplikatif, khususnya dengan membandingkan model machine learning tabular dan menganalisis faktor yang paling memengaruhi kelas salary.

## 8. Narasi Siap Pakai untuk Tinjauan Pustaka

Narasi yang dapat dipakai atau disesuaikan:

Penelitian terdahulu menunjukkan bahwa salary analytics telah dikaji melalui beberapa arah utama. Pertama, studi salary prediction berbasis regresi berupaya memperkirakan nilai salary kontinu menggunakan model statistik dan machine learning. Kedua, studi income classification memanfaatkan algoritma klasifikasi untuk memetakan individu ke kelas pendapatan tertentu, tetapi umumnya masih terbatas pada target biner. Ketiga, beberapa penelitian mulai meninjau salary grading atau classification, namun belum konsisten dalam menekankan evaluasi komparatif model tabular yang praktis untuk data campuran numerik dan kategorikal. Berdasarkan kondisi tersebut, penelitian ini menempatkan diri pada formulasi multiclass salary range classification dengan fokus pada perbandingan model machine learning dan interpretasi faktor dominan dalam konteks HR analytics.

## 9. Referensi Inti yang Disarankan untuk Dicari dan Dilengkapi

Prioritas referensi untuk finalisasi daftar pustaka:

1. Meng et al. (2018) tentang intelligent salary benchmarking,
2. Matbouli dan Alghamdi (2022) tentang salary regression berbasis machine learning,
3. Wang et al. (2022) tentang faktor yang memengaruhi starting salary,
4. Kuo et al. (2021) tentang salary grading prediction,
5. Mamidala et al. (2024) tentang XGBoost untuk salary or income prediction,
6. Stow (2025) tentang explainable income prediction,
7. paper asli CatBoost,
8. paper asli XGBoost,
9. referensi SHAP jika explainability dipakai.

## 10. Kesimpulan Praktis

Matriks ini mendukung arah penelitian Anda secara cukup kuat. Anda tidak perlu mengklaim bahwa topik ini sepenuhnya baru. Yang perlu ditekankan adalah bahwa penelitian Anda menawarkan formulasi yang lebih aplikatif, yaitu multiclass salary range classification berbasis model tabular dengan fokus pada evaluasi performa dan interpretasi hasil untuk HR analytics.