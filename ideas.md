“Multi-Class Salary Range Classification Using Machine Learning Techniques”

1. Latar Belakang Penelitian

Sebagian besar penelitian salary prediction menggunakan pendekatan:

regression,
menghasilkan angka salary exact.

Masalahnya:

di dunia nyata HR/recruitment,
perusahaan sering tidak membutuhkan angka exact,
tetapi kategori salary:
entry level,
middle,
senior,
premium.

Contoh nyata:

rekomendasi salary kandidat,
filtering kandidat,
benchmarking HR,
job recommendation system,
compensation analytics.

Karena itu, pendekatan:

Multi-Class Salary Classification

lebih realistis dan lebih aplikatif dibanding regression murni.

2. Ide Penelitian
Konsep Utama

Mengubah:

salary = angka kontinu

menjadi:

LOW
MEDIUM
HIGH
PREMIUM

lalu membangun model klasifikasi machine learning untuk:

memprediksi kelas salary,
membandingkan performa beberapa algoritma,
menganalisis faktor paling berpengaruh.
3. Research Gap
Gap Umum

Sebagian besar paper:

fokus regression,
hanya mengejar MAE/RMSE,
minim explainability,
kurang relevan untuk decision support HR.
Gap yang Bisa Anda Ambil
Gap 1 — Regression Dominance

Penelitian salary prediction masih didominasi pendekatan regresi.

Gap 2 — Practical Classification

Belum banyak penelitian yang:

mengklasifikasikan salary range,
untuk kebutuhan HR analytics.
Gap 3 — Comparative Study

Belum ada evaluasi komprehensif:

XGBoost,
CatBoost,
ANN
untuk multi-class salary classification.
Gap 4 — Feature Impact

Belum dianalisis:

fitur apa paling mempengaruhi kelas salary.
4. Tujuan Penelitian
Tujuan Utama

Mengembangkan model klasifikasi multi-kelas untuk memprediksi kategori salary berdasarkan atribut pekerjaan dan profesional.

Tujuan Detail
Mengubah salary menjadi kelas kategori.
Membangun model:
XGBoost,
CatBoost,
ANN.
Membandingkan performa model.
Mengevaluasi model menggunakan:
F1-score,
ROC-AUC,
confusion matrix.
Mengidentifikasi faktor paling berpengaruh.
5. Konsep Salary Classification
Contoh Salary Bracket

Misalnya:

Salary Range	Class
< 50k	LOW
50k – 100k	MEDIUM
100k – 150k	HIGH
> 150k	PREMIUM
Alternatif Lebih Akademis

Gunakan:

quartile,
percentile.

Contoh:

Q1 = LOW
Q2 = MEDIUM
Q3 = HIGH
Q4 = PREMIUM
Kenapa lebih bagus?

Karena:

distribusi lebih seimbang,
mengurangi class imbalance,
lebih scientific.
6. Arsitektur Penelitian
Dataset
   ↓
Data Cleaning
   ↓
EDA
   ↓
Salary Categorization
   ↓
Encoding & Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ├── XGBoost
   ├── CatBoost
   └── ANN
   ↓
Evaluation
   ↓
Comparison & Analysis
7. Metodologi Penelitian
7.1 Data Preprocessing
Tahapan
missing value handling,
duplicate removal,
outlier checking,
label encoding,
one-hot encoding.
7.2 Feature Engineering
Feature yang Digunakan
experience_years
education_level
skills_count
industry
company_size
location
remote_work
certifications
job_title
7.3 Encoding
Untuk High Cardinality

Karena:

job_title,
location,
industry

punya banyak kategori.

Gunakan:

frequency encoding,
target encoding,
CatBoost encoding.
8. Model Machine Learning
A. XGBoost Classifier
Kelebihan
kuat untuk tabular data,
cepat,
feature importance bagus,
performa tinggi.
Hyperparameter
max_depth
learning_rate
n_estimators
subsample
B. CatBoost
Kelebihan
sangat bagus untuk categorical data,
preprocessing lebih minimal,
stabil terhadap overfitting.
Cocok untuk dataset ini

Karena banyak fitur kategorikal.

C. Artificial Neural Network (ANN)
Tujuan

Membandingkan:

traditional boosting
vs
deep learning.
Arsitektur Sederhana
Input Layer
↓
Dense(128)
↓
Dropout
↓
Dense(64)
↓
Dense(4 Softmax)
9. Evaluasi Model
A. Accuracy

Untuk overview umum.

B. F1-Score

Paling penting karena:

multiclass,
kemungkinan imbalance.

Gunakan:

macro F1,
weighted F1.
C. Confusion Matrix

Untuk melihat:

kelas mana paling sering salah.

Contoh:

HIGH → diprediksi MEDIUM
D. ROC-AUC Multiclass

Gunakan:

One-vs-Rest ROC.
E. Cross Validation

Gunakan:

Stratified K-Fold.

Ini meningkatkan kualitas penelitian.

10. Feature Importance Analysis

Bagian penting supaya paper tidak terasa “cuma modeling”.

Gunakan:
SHAP,
permutation importance,
built-in feature importance.
Insight yang Bisa Didapat

Contoh:

experience_years paling dominan,
remote_work meningkatkan peluang PREMIUM,
certifications membantu HIGH salary.
11. Research Contribution
Kontribusi Akademik
Pendekatan salary classification multi-kelas.
Comparative analysis antar model.
Analisis fitur paling berpengaruh.
Framework HR salary recommendation.
Kontribusi Praktis
membantu HR,
salary benchmarking,
rekomendasi karir,
compensation planning.
12. Roadmap Penelitian
Phase 1 — Literature Review
Target

Mencari:

salary prediction paper,
HR analytics,
multiclass classification,
explainable AI.
Output
research gap,
state of the art.
Phase 2 — Data Understanding & EDA
Aktivitas
distribusi salary,
distribusi job title,
korelasi feature,
imbalance checking.
Visualisasi
histogram,
boxplot,
heatmap,
violin plot.
Phase 3 — Salary Categorization
Aktivitas

Membuat:

LOW,
MEDIUM,
HIGH,
PREMIUM.
Rekomendasi

Gunakan quartile.

Phase 4 — Data Preprocessing
Aktivitas
encoding,
scaling,
split dataset,
balancing jika perlu.
Phase 5 — Baseline Modeling
Model
Logistic Regression,
Decision Tree.

Tujuan:

baseline comparison.
Phase 6 — Advanced Modeling
Implementasi
XGBoost
CatBoost
ANN
Phase 7 — Hyperparameter Tuning
Tools
Optuna,
RandomSearchCV.
Phase 8 — Evaluation
Bandingkan
Model	Accuracy	F1	ROC-AUC
Phase 9 — Explainability
Gunakan
SHAP summary plot,
feature importance,
dependence plot.
Phase 10 — Discussion

Bahas:

kenapa model tertentu unggul,
pengaruh feature,
implikasi HR.
Phase 11 — Conclusion & Future Work
Future Work
transformer tabular model,
salary recommendation system,
real-time recruitment analytics.
13. Potensi Hasil Penelitian

Kemungkinan:

CatBoost outperform karena categorical data.
XGBoost close second.
ANN belum tentu terbaik.

Ini justru menarik.

14. Nilai Tambah Agar Lebih Kuat

Tambahkan salah satu:

Opsi 1 — Explainable AI

SHAP.

Opsi 2 — Imbalanced Learning

SMOTE.

Opsi 3 — Optuna Hyperparameter Optimization
Opsi 4 — Feature Interaction

Contoh:

experience × education
15. Judul Alternatif yang Lebih “SINTA-Friendly”
Opsi 1

“Multi-Class Salary Range Classification Using Ensemble Machine Learning Techniques”

Opsi 2

“Comparative Analysis of Machine Learning Models for Salary Range Classification”

Opsi 3

“Explainable Multi-Class Salary Classification Using XGBoost, CatBoost, and Artificial Neural Networks”

16. Tingkat Kesulitan
Bagian	Difficulty
EDA	Easy
XGBoost	Easy
CatBoost	Easy
ANN	Medium
SHAP	Medium
Optuna	Medium