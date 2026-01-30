# Analysis Scripts Inventory

**Generated:** January 25, 2026 (Updated: January 27, 2026)  
**Purpose:** Drone binary classification analysis & audio dataset comparison tools

---

## 📊 DUPLICATE ANALYSIS

### Exact Duplicates Found:

| Group | Files | Status |
|-------|-------|--------|
| **Group A** | `1_basic_histogram.py`, `archive/1analysis.py.py`, `archive/OGanalysis.py.py`, `archive/From Model diagnostic/analysis.py.py` | ❌ 4 identical copies |
| **Group B** | `3_multi_dataset_comparison.py`, `archive/From Model diagnostic/newAnalysis.py` | ❌ 2 identical copies |
| **Group C** | `4_threshold_evaluator.py`, `archive/From Model diagnostic/UpdatedEvalScript.py` | ❌ 2 identical copies |
| **Group D** | `FromAnotherProject/analyze_audio.py`, `FromAnotherProject/analyzer2.py` | ⚠️ Similar purpose (analyzer2 is enhanced version) |
| **Group E** | `FromAnotherProject/divergences.py`, `FromAnotherProject/divergences2.py` | ⚠️ Similar purpose (divergences.py is more comprehensive) |

### Unique Scripts (No Duplicates):
- `2_batch_histogram_validator.py` - Recursive search + column validation
- `5_graph_file_organizer.py` - File organization utility
- `FromAnotherProject/analysis.py` - Multi-analysis suite (hist/ROC/DET/Bayes/CM)
- `FromAnotherProject/DetPlotter.py` - Multi-file DET comparison
- `FromAnotherProject/DetPlotterZoomed.py` - DET comparison with zoom inset

---

## 📁 SCRIPTS ORGANIZED BY FUNCTION

---

# 🎯 CATEGORY A: MODEL EVALUATION SCRIPTS
*For analyzing model prediction CSV files (drone_probability, true_label, etc.)*

---

### A1. BASIC HISTOGRAM GENERATION (Entry Level)
**Active Script:** `1_basic_histogram.py`  
**Archived:** `archive/1analysis.py.py`, `archive/OGanalysis.py.py`, `archive/From Model diagnostic/analysis.py.py`

**When to use:** Quick visualization of a single evaluation run

**Features:**
- Generates simple log-scale histograms
- Processes all CSV files in a directory
- Shows drone vs no-drone probability distributions
- Displays plots interactively with `plt.show()`

**Usage:** `python 1_basic_histogram.py [directory]`

---

### A2. BATCH HISTOGRAM WITH VALIDATION (Production)
**Active Script:** `2_batch_histogram_validator.py`

**When to use:** Processing many evaluation files, need validation

**Features:**
- **Recursive** CSV file discovery in subdirectories
- **Validates required columns** before processing:
  - `file_id`, `true_label`, `predicted_label`, `drone_probability`
  - `aggregation_method`, `aggregation_threshold`, `split`
- **Saves plots as PNG files** (doesn't display interactively)
- Includes processing summary with counts

**Usage:** `python 2_batch_histogram_validator.py [directory]`

---

### A3. MULTI-DATASET COMPARISON GRID
**Active Script:** `3_multi_dataset_comparison.py`  
**Archived:** `archive/From Model diagnostic/newAnalysis.py`

**When to use:** Comparing model performance across multiple test datasets

**Features:**
- Creates **grid of subplots** comparing multiple datasets
- Calculates **performance metrics**: Accuracy, F1, Precision, Recall
- Ranks datasets by accuracy
- Provides comprehensive summary table
- Supports both combined and individual plot modes

**Usage:** `python 3_multi_dataset_comparison.py [directory]`

---

### A4. THRESHOLD EVALUATOR WITH F1 SCORING
**Active Script:** `4_threshold_evaluator.py`  
**Archived:** `archive/From Model diagnostic/UpdatedEvalScript.py`

**When to use:** Finding optimal decision threshold, threshold tuning

**Features:**
- **Custom probability threshold** (default: 0.006, interactive input)
- Uses **sklearn** for F1 score calculation
- Reports accuracy and error rates at custom thresholds
- Useful for threshold optimization

**Usage:** `python 4_threshold_evaluator.py [directory]`

---

### A5. COMPREHENSIVE ANALYSIS SUITE ⭐ (RECOMMENDED)
**Script:** `FromAnotherProject/analysis.py` → **Rename to:** `6_comprehensive_evaluator.py`

**When to use:** Full model evaluation with multiple plot types

**Features:**
- **Interactive menu** to choose analysis type:
  - `hist` - Probability histogram (with threshold from CSV)
  - `roc` - ROC curve with AUC
  - `det` - DET curve (probit scale)
  - `bayes` - Normalized Bayes error plot
  - `cm` - Confusion matrix heatmap
- Recursive CSV discovery with column validation
- Reads threshold from `aggregation_threshold` column in CSV
- **All-in-one solution** for model diagnostics

**Required CSV columns:** `file_id`, `true_label`, `predicted_label`, `drone_probability`, `aggregation_method`, `aggregation_threshold`, `split`

**Usage:** `python 6_comprehensive_evaluator.py [directory]`

---

### A6. DET CURVE COMPARISON (Multi-Model)
**Script:** `FromAnotherProject/DetPlotter.py` → **Rename to:** `7_det_curve_comparison.py`

**When to use:** Comparing DET curves from multiple augmentation methods or model variants

**Features:**
- Plots **multiple DET curves** on same axes
- Parses epoch and augmentation info from filenames
- Probit scale axes (standard DET format)
- Finnish language labels (can be modified)
- Supports wav2vec2 evaluation file naming conventions

**Usage:** Place CSV files in directory, run `python 7_det_curve_comparison.py`

---

### A7. DET CURVE COMPARISON WITH ZOOM
**Script:** `FromAnotherProject/DetPlotterZoomed.py` → **Rename to:** `8_det_curve_zoomed.py`

**When to use:** Detailed DET comparison when curves are very close together

**Features:**
- Main DET plot with **zoomed inset**
- Computes **EER (Equal Error Rate)** for each model
- Computes **AUC** scores
- Generates **metrics comparison table**
- Saves metrics to CSV file
- Pairwise EER difference calculations

**Usage:** Place CSV files in directory, run `python 8_det_curve_zoomed.py`

---

# 🔊 CATEGORY B: AUDIO DATASET ANALYSIS SCRIPTS
*For analyzing raw audio file properties across datasets*

---

### B1. AUDIO DATASET ANALYZER (Basic)
**Script:** `FromAnotherProject/analyze_audio.py` → **Archive** (superseded by analyzer2.py)

**Features:**
- Analyzes audio files: length, RMS, peak, spectrum
- Generates overlaid histograms
- Generates boxplots
- Outputs CSV with per-file statistics

---

### B2. AUDIO DATASET ANALYZER (Enhanced) ⭐
**Script:** `FromAnotherProject/analyzer2.py` → **Rename to:** `9_audio_dataset_analyzer.py`

**When to use:** Comparing audio characteristics across training/test datasets

**Features:**
- **All features from analyze_audio.py PLUS:**
- **Crest factor** metric (peak/RMS ratio)
- **Per-dataset histograms** (organized in subfolders)
- Better code structure with type hints
- Configurable metrics system
- Total duration summary

**Metrics analyzed:** Duration, RMS, Peak, Crest Factor, Mean Spectrum

**Output structure:**
```
audio_analysis_results/
├── audio_dataset_stats_detailed.csv
├── audio_dataset_length_summary.csv
├── histograms/
│   ├── Dataset1/
│   │   ├── hist_length.png
│   │   ├── hist_rms.png
│   │   └── ...
│   └── Dataset2/
└── boxplots/
    ├── box_length.png
    └── ...
```

**Usage:** Configure `COMPARISON_DATASETS` dict, run `python 9_audio_dataset_analyzer.py`

---

### B3. DATASET DIVERGENCE ANALYSIS ⭐
**Script:** `FromAnotherProject/divergences.py` → **Rename to:** `10_dataset_divergence.py`

**When to use:** Measuring statistical similarity/difference between audio datasets

**Features:**
- **Jensen-Shannon divergence** calculation between all dataset pairs
- **Heatmap visualization** with hierarchical clustering
- **MDS (Multidimensional Scaling)** visualization
- Freedman-Diaconis binning for histograms
- Configurable display names for datasets (Finnish translations included)
- Handles missing data gracefully

**Metrics compared:** length, RMS, peak, spectrum, crest_factor

**Output:** Heatmaps and MDS plots in `audio_analysis_results/divergences/`

**Usage:** Requires `audio_analysis_results/audio_dataset_stats_detailed.csv` from analyzer2.py

---

### B4. DATASET VARIABILITY ANALYSIS (Simpler)
**Script:** `FromAnotherProject/divergences2.py` → **Archive** (simpler version of B3)

**Features:**
- Bar charts of variability measures (Std Dev, IQR, Coefficient of Variation)
- Less comprehensive than divergences.py

---

# 🛠️ CATEGORY C: UTILITY SCRIPTS

---

### C1. GRAPH FILE ORGANIZER
**Active Script:** `5_graph_file_organizer.py`

**When to use:** Organizing scattered histogram PNG files from batch processing

**Features:**
- Finds all `*_histogram.png` files recursively
- Copies with meaningful names based on path structure
- Creates file mapping summary document
- Organizes output into structured folders

**Usage:** `python 5_graph_file_organizer.py [source_directory]`

---

# 📂 CATEGORY D: MODEL/DATASET/EPOCH ANALYSIS SCRIPTS (ModelDatasetEpochAnalysis/)
*For analyzing multiple models across different epochs and datasets - Cross-model comparison*

---

## 📊 DUPLICATE/SIMILARITY ANALYSIS (ModelDatasetEpochAnalysis/)

| Group | Files | Status |
|-------|-------|--------|
| **Group F** | `archive/dataset_epoch_analysis.py`, `archive/dataset_epoch_analysis_separated.py`, `archive/dataset_epoch_analysis_weighted.py` | ✅ Archived - `4_dataset_epoch_weighted.py` is active |
| **Group G** | `archive/analysis.py`, `archive/eiRecursive/newAnalysis.py` | ✅ Archived - `6_basic_histogram.py` is active |
| **Group H** | `archive/forTransformer_Weighted.py`, `4_dataset_epoch_weighted.py` | ✅ `5_transformer_weighted.py` active (Transformer-specific) |
| **Group I** | `archive/working/test_*.py` files | ✅ Development/test files archived |

---

### D1. FUSION DATASET CREATOR ⭐ (RECOMMENDED FIRST STEP)
**Active Script:** `ModelDatasetEpochAnalysis/1_create_fusion_dataset.py`  
**Archived:** `archive/working/create_fusion_dataset.py`

**When to use:** First step - combine all datasets into one Fusion dataset per model/epoch for overall analysis

**Features:**
- Discovers all `*_detailed_files.csv` files recursively
- Groups by model, epoch, and folder structure (clustered vs distributed layouts)
- **Excludes** CalibrationDataset and existing Fusion datasets
- **Creates backups** of existing Fusion files before overwriting
- Supports both wav2vec2 and CNN-LSTM directory structures
- Adds `source_dataset` column for traceability

**Usage:** `python 1_create_fusion_dataset.py <root_directory> [output_directory] [backup_directory]`

**Example:**
```bash
python 1_create_fusion_dataset.py Z:\Experiment\Evaluation
python 1_create_fusion_dataset.py Z:\Experiment\Evaluation C:\Output C:\Backups
```

---

### D2. MODEL RANKING GENERATOR ⭐ (RECOMMENDED)
**Active Script:** `ModelDatasetEpochAnalysis/2_model_ranker.py`  
**Archived:** `archive/working/ranker4.py`

**When to use:** After fusion - rank models by performance (Brier Score, F1)

**Features:**
- Extracts model, epoch, and dataset info from file paths
- Calculates **Brier Score** (works for all datasets including single-class)
- Calculates **Precision, Recall, F1, Accuracy** (for two-class datasets)
- Finds **optimal threshold** per dataset (maximizes F1 or minimizes Brier)
- Compares **calibrated vs optimized thresholds**
- Creates **summary ranking table** sorted by Weighted Brier Score
- Creates **detailed per-dataset metrics table**
- Outputs threshold comparison analysis

**Output files:**
- `model_ranking_summary_calibrated.csv` - Overall model rankings (calibrated threshold)
- `model_ranking_summary_optimized.csv` - Rankings with dataset-optimized thresholds
- `model_ranking_detailed_calibrated.csv` - Per-dataset breakdown
- `threshold_comparison.csv` - Calibrated vs optimized threshold analysis

**Usage:** `python 2_model_ranker.py <root_directory>`

---

### D3. EPOCH STATISTICS GENERATOR (Comprehensive) Needs Rework for updated file structure
**Active Script:** `ModelDatasetEpochAnalysis/3_epoch_statistics.py`  
**Archived:** `archive/statistics.py`

**When to use:** Generate detailed statistical summaries per epoch with LaTeX output

**Features:**
- Computes **per-class probability statistics** (Drone vs No-Drone):
  - Mean, Median, Min, Max, Q1, Q3, Std, 95% CI
- Calculates **separation metrics**: Mean Delta, Mean Ratio, Cohen's d
- Computes **PR-AUC (Average Precision)** and **ROC-AUC**
- Generates **PR and ROC curves** as PNG files
- Outputs **LaTeX tables** for academic papers
- Creates master LaTeX document including all epoch tables

**Output structure:**
```
EpochStats/
├── Epoch1Stats.csv
├── Epoch5Stats.csv
├── Epoch1Stats.tex
├── Epoch5Stats.tex
├── EpochStats_All.tex (master document)
├── PR_Curves/
│   └── Epoch1_DatasetName_PRCurve.png
└── ROC_Curves/
    └── Epoch1_DatasetName_ROCCurve.png
```

**Usage:** `python 3_epoch_statistics.py <root_directory> [--debug]`

---

### D4. DATASET EPOCH VISUALIZATION (Weighted) ⭐
**Active Script:** `ModelDatasetEpochAnalysis/4_dataset_epoch_weighted.py`  
**Archived:** `archive/dataset_epoch_analysis_weighted.py`

**When to use:** Visualize probability distributions with density weighting

**Features:**
- **Separates by true_label**: Drone (1) vs Unknown/No-Drone (0)
- **Density-weighted visualization**: Line thickness/opacity varies based on data concentration
- **Median-based** statistics (more robust to outliers than mean)
- Histogram-based density calculation (configurable bins)
- Generates plots showing min-median-max ranges

**This is the most advanced version of the epoch analysis scripts.**

**Usage:** `python 4_dataset_epoch_weighted.py <root_directory>`

---

### D5. TRANSFORMER-SPECIFIC ANALYSIS
**Active Script:** `ModelDatasetEpochAnalysis/5_transformer_weighted.py`  
**Archived:** `archive/forTransformer_Weighted.py`

**When to use:** Specifically for wav2vec2/Transformer model evaluation results

**Features:**
- **KDE (Kernel Density Estimation)** for smooth density curves
- Robust dataset/epoch extraction from complex path structures
- Handles evaluation folder naming patterns
- Reads threshold from `aggregation_threshold` column

**Similar to D4 but with different density calculation method (KDE vs histogram).**

---

### D6. BASIC HISTOGRAM GENERATOR
**Active Script:** `ModelDatasetEpochAnalysis/6_basic_histogram.py`  
**Archived:** `archive/analysis.py`

**When to use:** Quick histogram generation for prediction files

**Features:**
- Log-scale histogram visualization
- Adaptive bin count based on sample size
- Validates required columns
- Saves histograms as PNG files alongside CSV

**Note:** Simpler version than the main `1_basic_histogram.py` in root folder.

---

### D7. GRAPH FILE ORGANIZER
**Active Script:** `ModelDatasetEpochAnalysis/7_organize_graphs.py`  
**Archived:** `archive/organize_graphs.py`

**When to use:** After generating many histogram PNG files

**Features:**
- Finds all `*_histogram.png` files recursively
- Creates meaningful names based on path structure
- Filters out timestamp-based directory names
- Copies to organized output directory

**Usage:** `python 7_organize_graphs.py <source_directory> [output_directory]`

---

### D8. AUDIO FILE CALCULATOR (Utility)
**Active Script:** `ModelDatasetEpochAnalysis/utils/audio_calculator.py`  
**Archived:** `archive/calculate.py`

**When to use:** Calculate audio file durations and statistics

**Features:**
- Uses librosa for audio file processing
- Calculates duration, sample rate, channels, format
- Directory-based statistics aggregation
- **Not related to model evaluation** - utility for audio dataset inspection

**Note:** This is a general utility, not part of the evaluation workflow.

---

## ✅ ARCHIVED SCRIPTS (ModelDatasetEpochAnalysis/archive/)

### Scripts Now in Archive:

| Script | Reason |
|--------|--------|
| `archive/dataset_epoch_analysis.py` | Basic version - superseded by `4_dataset_epoch_weighted.py` |
| `archive/dataset_epoch_analysis_separated.py` | Intermediate version - superseded by `4_dataset_epoch_weighted.py` |
| `archive/eiRecursive/newAnalysis.py` | Older grid histogram generator with hardcoded threshold |
| `archive/working/SwapTrueLabel.py` | **DO NOT USE** - Dangerous label manipulation tool |
| `archive/working/test_*.py` | Development test files |
| `archive/toFix/cnn_lstm_evaluation.py` | Model evaluation code needing fixes |
| `archive/toFix/cnn_lstm_model.py` | Model definition code needing fixes |

---

## 🚀 RECOMMENDED WORKFLOW: ModelDatasetEpochAnalysis

```
STEP 1: Create Fusion Datasets (combines all evaluation results)
┌─────────────────────────────────────────────────────────────────┐
│  python 1_create_fusion_dataset.py Z:\Experiment\Evaluation     │
│  → Creates Fusion_detailed_files.csv for each model/epoch       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 2: Generate Model Rankings
┌─────────────────────────────────────────────────────────────────┐
│  python 2_model_ranker.py Z:\Experiment\Evaluation              │
│  → Creates model_ranking_summary.csv (sorted by Brier Score)    │
│  → Compares calibrated vs optimized thresholds                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 3: Generate Epoch Statistics (optional - for papers)
┌─────────────────────────────────────────────────────────────────┐
│  python 3_epoch_statistics.py Z:\Experiment\Evaluation          │
│  → Creates per-epoch CSV with detailed stats                    │
│  → Creates LaTeX tables for academic papers                     │
│  → Generates PR and ROC curve plots                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 4: Visualize Dataset/Epoch Distributions (optional)
┌─────────────────────────────────────────────────────────────────┐
│  python 4_dataset_epoch_weighted.py Z:\Experiment\Evaluation    │
│  → Creates density-weighted probability distribution plots      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 5: Organize Output Files
┌─────────────────────────────────────────────────────────────────┐
│  python 7_organize_graphs.py Z:\Experiment\Evaluation           │
│  → Collects and renames all generated PNG files                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 RECOMMENDED FOLDER STRUCTURE (ModelDatasetEpochAnalysis/ - After Reorganization)

```
ModelDatasetEpochAnalysis/
│
├── ── MAIN WORKFLOW SCRIPTS ──
├── 1_create_fusion_dataset.py     # Step 1: Combine datasets
├── 2_model_ranker.py              # Step 2: Rank models by Brier/F1
├── 3_epoch_statistics.py          # Step 3: Detailed stats + LaTeX
├── 4_dataset_epoch_weighted.py    # Step 4: Distribution visualization
├── 5_transformer_weighted.py      # Alternative: Transformer-specific
├── 6_basic_histogram.py           # Quick histograms
├── 7_organize_graphs.py           # Organize output files
│
├── utils/                         # Utility scripts
│   └── audio_calculator.py        # Audio file stats (not evaluation)
│
├── requirements.txt               # Dependencies
│
└── archive/                       # Deprecated/superseded scripts
    ├── dataset_epoch_analysis.py
    ├── dataset_epoch_analysis_separated.py
    ├── SwapTrueLabel.py           # DO NOT USE
    ├── eiRecursive/
    │   └── newAnalysis.py
    ├── toFix/
    │   ├── cnn_lstm_evaluation.py
    │   └── cnn_lstm_model.py
    └── working_tests/
        ├── test_backup_naming.py
        ├── test_final_logic.py
        └── ...other test files
```

---

## 🔄 QUICK REFERENCE: ModelDatasetEpochAnalysis Scripts

| I want to... | Use this script |
|--------------|-----------------|
| **Combine all datasets into one** | `1_create_fusion_dataset.py` |
| **Rank models by overall performance** | `2_model_ranker.py` |
| **Generate detailed statistics + LaTeX** | `3_epoch_statistics.py` |
| **Visualize probability distributions** | `4_dataset_epoch_weighted.py` |
| **Analyze Transformer model results** | `5_transformer_weighted.py` |
| **Quick histogram of a CSV file** | `6_basic_histogram.py` |
| **Organize scattered PNG files** | `7_organize_graphs.py` |
| **Calculate audio file durations** | `utils/audio_calculator.py` |

---

## 🗂️ RECOMMENDED FOLDER STRUCTURE (Root)

```
All Analysis Scripts/
│
├── SCRIPT_INVENTORY.md            # This documentation file
│
├── ── MODEL EVALUATION ──
├── 1_basic_histogram.py           # Quick single-run visualization
├── 2_batch_histogram_validator.py # Production batch processing
├── 3_multi_dataset_comparison.py  # Grid comparison across datasets
├── 4_threshold_evaluator.py       # Threshold optimization
├── 6_comprehensive_evaluator.py   # Full suite (hist/ROC/DET/Bayes/CM) ⭐
├── 7_det_curve_comparison.py      # Multi-model DET comparison
├── 8_det_curve_zoomed.py          # DET with zoom + metrics table
│
├── ── AUDIO ANALYSIS ──
├── 9_audio_dataset_analyzer.py    # Audio file statistics ⭐
├── 10_dataset_divergence.py       # JS divergence heatmaps ⭐
│
├── ── UTILITIES ──
├── 5_graph_file_organizer.py      # Organize output files
│
├── archive/                       # Old/duplicate scripts
│   ├── 1analysis.py.py
│   ├── OGanalysis.py.py
│   ├── From Model diagnostic/
│   │   ├── analysis.py.py
│   │   ├── newAnalysis.py
│   │   └── UpdatedEvalScript.py
│   └── FromAnotherProject/
│       ├── analysis.py            # Original (copied to 6_)
│       ├── analyzer2.py           # Original (copied to 9_)
│       ├── analyze_audio.py       # Superseded by analyzer2.py
│       ├── DetPlotter.py          # Original (copied to 7_)
│       ├── DetPlotterZoomed.py    # Original (copied to 8_)
│       ├── divergences.py         # Original (copied to 10_)
│       └── divergences2.py        # Superseded by divergences.py
│
└── FromAnotherProject/            # Output data files (not scripts)
    ├── *.csv                      # Analysis results
    ├── *.png                      # Generated plots
    ├── audio_analysis_results/    # Audio analysis outputs
    ├── boxplots/
    ├── divergences/
    └── histograms/
```

---

## 🚀 QUICK REFERENCE: WHICH SCRIPT TO USE?

| I want to... | Use this script |
|--------------|-----------------|
| **Quickly visualize one CSV file** | `1_basic_histogram.py` |
| **Process many CSV files with validation** | `2_batch_histogram_validator.py` |
| **Compare performance across datasets** | `3_multi_dataset_comparison.py` |
| **Find the optimal decision threshold** | `4_threshold_evaluator.py` |
| **Full analysis with ROC/DET/Bayes plots** | `6_comprehensive_evaluator.py` ⭐ |
| **Compare DET curves for multiple models** | `7_det_curve_comparison.py` |
| **Detailed DET comparison with EER metrics** | `8_det_curve_zoomed.py` |
| **Analyze audio file characteristics** | `9_audio_dataset_analyzer.py` ⭐ |
| **Measure dataset similarity/divergence** | `10_dataset_divergence.py` ⭐ |
| **Organize scattered output PNG files** | `5_graph_file_organizer.py` |

---

## 🔄 RECOMMENDED WORKFLOW FOR NEW MODEL EVALUATION

```
1. Run evaluation → Get CSV files with predictions
                    ↓
2. python 6_comprehensive_evaluator.py [dir]
   → Choose: hist, roc, det, bayes, or cm
   → Get individual analysis plots
                    ↓
3. python 3_multi_dataset_comparison.py [dir]
   → Get side-by-side comparison grid
                    ↓
4. python 4_threshold_evaluator.py [dir]
   → Optimize decision threshold
                    ↓
5. python 5_graph_file_organizer.py [dir]
   → Organize all generated PNG files
```

---

## 🔊 WORKFLOW FOR AUDIO DATASET ANALYSIS

```
1. Configure COMPARISON_DATASETS in 9_audio_dataset_analyzer.py
                    ↓
2. python 9_audio_dataset_analyzer.py
   → Generates: per-file stats CSV, histograms, boxplots
                    ↓
3. python 10_dataset_divergence.py
   → Reads stats CSV from step 2
   → Generates: JS divergence heatmaps, MDS plots
```

---

## ⚠️ NOTES

- Scripts 1-5 are the **original organized scripts** from this folder
- Scripts 6-10 are **newly added** from FromAnotherProject
- All duplicates have been moved to `archive/`
- The `FromAnotherProject/` folder now only contains **output data** (CSV, PNG files)
- ⭐ marks the **recommended** script for each category
