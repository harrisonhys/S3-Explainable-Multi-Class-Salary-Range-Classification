# Standar Output dan Penyimpanan Hasil per Fase CRISP-DM

## 1. Tujuan Dokumen

Dokumen ini berfungsi sebagai standar penyimpanan hasil agar setiap fase penelitian menghasilkan artefak yang seragam, mudah dicek, dan mudah dipakai kembali saat penulisan artikel.

## 2. Root Folder Hasil

Seluruh output penelitian disimpan di folder:

`crispdm_results/`

Struktur utama:

1. `phase1_business_understanding/`
2. `phase2_data_understanding/`
3. `phase3_data_preparation/`
4. `phase4_modeling/`
5. `phase5_evaluation/`
6. `phase6_deployment_research/`
7. `final_conclusion/`

Setiap phase minimal memiliki subfolder:

1. `tables/`
2. `plots/`
3. `charts/`
4. `insights/`
5. `results.txt`

## 3. Aturan Isi Folder

## 3.1 tables/

Isi folder ini berupa file `.txt` yang berisi tabel atau ringkasan numerik. Contoh:

1. `dataset_overview.txt`
2. `class_distribution.txt`
3. `model_comparison.txt`
4. `feature_importance.txt`

## 3.2 plots/

Isi folder ini berupa visual berbasis statistik atau analitik. Contoh:

1. histogram,
2. boxplot,
3. heatmap,
4. confusion matrix,
5. feature importance plot.

## 3.3 charts/

Isi folder ini berupa visual ringkas atau diagram presentasional. Contoh:

1. bar chart ranking model,
2. pie chart distribusi fitur,
3. comparison chart,
4. flow chart,
5. summary chart.

## 3.4 insights/

Isi folder ini berupa file teks insight singkat yang benar-benar berguna untuk keputusan fase berikutnya.

## 3.5 results.txt

File ini adalah ringkasan hasil utama per fase. File ini wajib ada di setiap fase.

## 4. Template Isi results.txt

Format minimal:

```text
PHASE:
Tanggal:

RINGKASAN HASIL:
-
-
-

TEMUAN UTAMA:
-
-

KEPUTUSAN LANJUT:
- lanjut ke fase berikutnya / revisi fase sebelumnya

CATATAN RISIKO:
-
```

## 5. Template Isi insight

Format minimal:

```text
INSIGHT PHASE X

Temuan:
-

Makna:
-

Dampak pada fase berikutnya:
-
```

## 6. Naming Convention yang Disarankan

Gunakan penamaan file yang konsisten, misalnya:

1. `01_dataset_overview.txt`
2. `02_missing_values.txt`
3. `03_class_distribution.txt`
4. `01_salary_distribution.png`
5. `02_confusion_matrix.png`
6. `01_model_ranking_chart.png`

Keuntungan:

1. artefak lebih mudah dibaca berurutan,
2. lebih mudah dipindahkan ke laporan atau artikel,
3. lebih mudah dicek jika ada file yang belum dibuat.

## 7. Checklist Kelengkapan per Fase

Sebelum pindah fase, cek apakah item berikut sudah ada:

1. minimal satu file tabel `.txt`,
2. minimal satu plot,
3. minimal satu chart,
4. `results.txt`,
5. insight jika memang ada temuan penting.

## 8. Output Minimum per Phase

## Phase 1

1. business objectives table,
2. research flow chart,
3. results.txt.

## Phase 2

1. dataset overview table,
2. missing values table,
3. salary distribution plot,
4. feature type chart,
5. results.txt.

## Phase 3

1. class distribution table,
2. preprocessing summary table,
3. class distribution plot,
4. preprocessing chart,
5. results.txt.

## Phase 4

1. baseline metrics table,
2. model comparison table,
3. metric comparison plot,
4. model ranking chart,
5. results.txt.

## Phase 5

1. final comparison table,
2. classification report table,
3. confusion matrix plot,
4. performance ranking chart,
5. feature importance output,
6. results.txt.

## Phase 6

1. daftar aset untuk paper,
2. tabel final siap artikel,
3. chart final,
4. results.txt.

## Final Conclusion

1. jawaban rumusan masalah,
2. ringkasan kontribusi,
3. conclusion chart,
4. results.txt.

## 9. Aturan Kualitas Dokumentasi

Setiap hasil yang disimpan harus:

1. dapat dibaca tanpa membuka notebook,
2. memiliki nama file yang jelas,
3. punya konteks singkat di `results.txt`,
4. cukup kuat untuk dipakai langsung pada penulisan hasil dan pembahasan.

## 10. Kesimpulan Praktis

Dengan standar ini, setiap fase CRISP-DM tidak hanya menghasilkan proses, tetapi juga jejak hasil yang rapi. Ini penting untuk:

1. memudahkan evaluasi progres penelitian,
2. memudahkan reproduksibilitas,
3. mempercepat penulisan artikel,
4. memastikan fase kesimpulan dibangun dari hasil yang benar-benar terdokumentasi.
