# Pipeline CRISP-DM - Panduan Eksekusi

## Ringkasan Singkat

`pipeline_crispdm.py` adalah script otomatis yang menjalankan **7 fase CRISP-DM** untuk penelitian salary range classification. Script ini:

- ✅ Download/load dataset otomatis
- ✅ Eksekusi semua 7 fase secara berurutan
- ✅ Simpan output ke folder terstruktur (crispdm_results/)
- ✅ Generate tables, plots, charts untuk publikasi
- ✅ Enforce dependency antar fase
- ✅ Dokumentasi lengkap per fase

## Prasyarat

### 1. Python Environment
```bash
# Pastikan Python >= 3.9 terinstall
python --version

# Recommended: Gunakan virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements-pipeline.txt
```

## Cara Menjalankan Pipeline

### Metode 1: Direct Execution (Recommended)
```bash
cd /Users/macbook/Documents/PERSONAL/Data Science/"S3 Multilabel Sallary"
python pipeline_crispdm.py
```

Expected output:
```
================================================================================
STARTING CRISP-DM PIPELINE FOR SALARY RANGE CLASSIFICATION RESEARCH
================================================================================
Started at: 2024-XX-XX XX:XX:XX

================================================================================
PHASE 1: BUSINESS UNDERSTANDING
================================================================================
✓ Saved: .../phase1_business_understanding/tables/business_objectives.txt
✓ Saved: .../phase1_business_understanding/tables/research_questions.txt
...
[More output for each phase]
...
================================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY
================================================================================
Best Model: CatBoost
Best F1 (Macro): 0.7234

Results saved to: .../crispdm_results/
```

### Metode 2: Run with Output Logging
```bash
python pipeline_crispdm.py 2>&1 | tee pipeline_execution.log
```

## Output Structure

Setelah pipeline selesai, struktur folder akan terlihat seperti:

```
S3 Multilabel Sallary/
├── crispdm_results/
│   ├── README.md (Overview)
│   │
│   ├── phase1_business_understanding/
│   │   ├── tables/
│   │   │   ├── business_objectives.txt
│   │   │   └── research_questions.txt
│   │   ├── plots/
│   │   ├── charts/
│   │   │   └── research_flow_chart.txt
│   │   ├── insights/
│   │   ├── logs/
│   │   │   └── phase1_business_understanding.log
│   │   └── results.txt
│   │
│   ├── phase2_data_understanding/
│   │   ├── tables/
│   │   │   ├── dataset_overview.txt
│   │   │   ├── missing_values.txt
│   │   │   ├── categorical_cardinality.txt
│   │   │   └── target_variable_analysis.txt
│   │   ├── plots/
│   │   │   └── salary_distribution.png
│   │   ├── charts/
│   │   ├── insights/
│   │   ├── logs/
│   │   └── results.txt
│   │
│   ├── phase3_data_preparation/
│   │   ├── tables/
│   │   │   ├── target_class_distribution.txt
│   │   │   └── data_splitting_summary.txt
│   │   ├── plots/
│   │   │   └── class_distribution.png
│   │   ├── charts/
│   │   ├── insights/
│   │   ├── logs/
│   │   └── results.txt
│   │
│   ├── phase4_modeling/
│   │   ├── tables/
│   │   │   └── training_log.txt
│   │   ├── plots/
│   │   │   └── feature_importance.png
│   │   ├── charts/
│   │   ├── insights/
│   │   ├── logs/
│   │   └── results.txt
│   │
│   ├── phase5_evaluation/
│   │   ├── tables/
│   │   │   ├── model_comparison.txt
│   │   │   └── best_model_classification_report.txt
│   │   ├── plots/
│   │   │   ├── confusion_matrix_best_model.png
│   │   │   └── model_comparison_chart.png
│   │   ├── charts/
│   │   ├── insights/
│   │   ├── logs/
│   │   └── results.txt
│   │
│   ├── phase6_deployment_research/
│   │   ├── tables/
│   │   │   ├── paper_ready_model_comparison.txt
│   │   │   └── research_findings_summary.txt
│   │   ├── plots/
│   │   ├── charts/
│   │   ├── insights/
│   │   ├── logs/
│   │   └── results.txt
│   │
│   └── phase7_final_conclusion/
│       ├── tables/
│       │   ├── research_questions_answers.txt
│       │   ├── research_contribution.txt
│       │   ├── research_limitations.txt
│       │   ├── future_work_recommendations.txt
│       │   └── final_synthesis.txt
│       ├── plots/
│       ├── charts/
│       ├── insights/
│       ├── logs/
│       └── results.txt
│
├── datasets/
│   └── ds_salaries.csv (auto-downloaded if not exist)
│
├── pipeline_crispdm.py (Main script)
├── requirements-pipeline.txt
└── ... (other project files)
```

## Fase-Fase dalam Pipeline

### Phase 1: Business Understanding
- **Durasi**: ~30 detik
- **Output**: Tujuan penelitian, research questions, success criteria
- **Artefak**: business_objectives.txt, research_questions.txt

### Phase 2: Data Understanding
- **Durasi**: ~1-2 menit
- **Output**: Dataset overview, EDA, quality assessment
- **Artefak**: dataset_overview.txt, missing_values.txt, categorical_cardinality.txt, salary_distribution.png
- **Fallback**: Jika dataset tidak ditemukan, auto-generate sample data (untuk development/testing)

### Phase 3: Data Preparation
- **Durasi**: ~1-2 menit  
- **Output**: Feature preprocessing, target discretization (4-class), data splitting
- **Artefak**: target_class_distribution.txt, data_splitting_summary.txt, class_distribution.png
- **Key**: Stratified split mempertahankan class balance

### Phase 4: Modeling
- **Durasi**: ~2-3 menit
- **Output**: Training 3 models (Logistic Regression, XGBoost, CatBoost)
- **Artefak**: training_log.txt, feature_importance.png
- **Models**:
  - Logistic Regression (baseline)
  - XGBoost (gradient boosting)
  - CatBoost (categorical optimization)

### Phase 5: Evaluation
- **Durasi**: ~1 menit
- **Output**: Comprehensive model evaluation & comparison
- **Metrics**: Accuracy, F1 (Macro), F1 (Weighted), Precision, Recall
- **Artefak**: model_comparison.txt, confusion_matrix.png, classification_report.txt
- **Best Model Selection**: Berdasarkan macro F1-score (highest wins)

### Phase 6: Deployment for Research Output
- **Durasi**: ~30 detik
- **Output**: Artefak siap publikasi
- **Artefak**: paper_ready_model_comparison.txt, research_findings_summary.txt
- **Tujuan**: Persiapan untuk submission ke jurnal Sinta 3

### Phase 7: Final Conclusion
- **Durasi**: ~30 detik
- **Output**: Sintesis kesimpulan penelitian dengan menjawab semua RQ
- **Artefak**: 
  - research_questions_answers.txt (menjawab RQ1-RQ4)
  - research_contribution.txt (kontribusi teoritis & praktis)
  - research_limitations.txt (keterbatasan penelitian)
  - future_work_recommendations.txt (saran penelitian lanjutan)
  - final_synthesis.txt (summary keseluruhan)

**Total Durasi**: ~6-10 menit untuk full pipeline execution

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pandas'"
**Solusi**: Install dependencies terlebih dahulu
```bash
pip install -r requirements-pipeline.txt
```

### Problem: "Failed to download dataset from Kaggle"
**Solusi**: Script sudah punya fallback. Akan auto-generate sample dataset untuk development.
Untuk gunakan real data dari Kaggle:
1. Setup Kaggle API credentials: `~/.kaggle/kaggle.json`
2. Re-run script: `python pipeline_crispdm.py`

### Problem: "PermissionError: [Errno 13]..." saat write files
**Solusi**: Pastikan folder `crispdm_results/` accessible (permission + free space)
```bash
chmod -R 755 crispdm_results/
```

### Problem: "Memory Error" atau script hang di phase 4 (modeling)
**Solusi**: Reduce dataset size atau model complexity di script (edit lines 486-492 untuk smaller iterations)

## Output Files untuk Publikasi

File-file yang paling penting untuk publikasi ke Sinta 3:

**From Phase 5 (Evaluation):**
- `model_comparison.txt` → Table 1 di paper
- `confusion_matrix_best_model.png` → Figure dalam methodology/results

**From Phase 6 (Deployment):**
- `paper_ready_model_comparison.txt` → Ready untuk paste ke paper
- `research_findings_summary.txt` → Main findings section

**From Phase 7 (Conclusion):**
- `research_questions_answers.txt` → Discussion section
- `research_contribution.txt` → Contribution paragraph
- `research_limitations.txt` → Limitations section
- `future_work_recommendations.txt` → Future work section

## Next Steps Setelah Pipeline Selesai

1. **Review Results**:
   - Baca `crispdm_results/phase7_final_conclusion/tables/final_synthesis.txt`
   - Check best model performance vs target metric (F1 >= 0.70)

2. **Integrate dengan Paper Draft**:
   - Copy findings dari `paper_ready_*.txt` ke `03_draft_penulisan_sinta3.md`
   - Embed plots/charts dari crispdm_results/ ke paper document
   - Update methodology section dengan actual results

3. **Fine-tuning (Optional)**:
   - Jika F1 < target, edit pipeline untuk:
     - Feature engineering enhancement
     - Hyperparameter tuning
     - Different discretization strategy

4. **Submission Preparation**:
   - Compile paper dengan integrated results
   - Create supplementary materials dengan code + crispdm_results folder
   - Ready for Sinta 3 journal submission

## Kustomisasi Pipeline

### Ubah Dataset
Edit lines 80-150 dalam `pipeline_crispdm.py`:
```python
# Ganti di function download_dataset()
dataset_path = "path/to/your/dataset.csv"
```

### Ubah Model Kandidat
Edit line 490-530 dalam `phase_4_modeling()`:
```python
# Tambah model baru (contoh: Random Forest)
rf_model = RandomForestClassifier(...)
models['RandomForest'] = rf_model
```

### Ubah Target Metric
Edit line 25-26:
```python
TARGET_METRIC = "macro_f1_score"  # or "accuracy", "weighted_f1"
TARGET_THRESHOLD = 0.70  # Ganti sesuai kebutuhan
```

## File Kontrol & Monitoring

Setiap fase simpan `.log` file di `{phase}/logs/` untuk monitoring & debugging:
- `phase1_business_understanding.log`
- `phase2_data_understanding.log`
- ... dst

Check logs jika ada error:
```bash
tail -f crispdm_results/phase{N}_{name}/logs/{phase_name}.log
```

## Contact & Support

Pipeline ini dikembangkan untuk proyek "Salary Range Classification for Sinta 3 Research".

Untuk customizations atau troubleshooting lebih lanjut, refer ke:
- Pipeline documentation: `06_pipeline_penelitian_crispdm.md`
- Output standards: `07_standar_output_crispdm.md`
- Dataset plan: `04_dataset_plan_penelitian.md`

---

**Status**: Ready for Production Use ✅

**Version**: 1.0  
**Last Updated**: 2024
