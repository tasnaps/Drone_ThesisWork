# Analysis Scripts Inventory

**Last Updated:** February 8, 2026  
**Purpose:** Drone binary classification analysis and audio dataset comparison tools

---

## Quick Reference

| I want to... | Use this script |
|--------------|-----------------|
| **Quickly visualize one CSV file** | `1_basic_histogram.py` |
| **Process many CSV files with validation** | `2_batch_histogram_validator.py` |
| **Compare performance across datasets** | `3_multi_dataset_comparison.py` |
| **Find the optimal decision threshold** | `4_threshold_evaluator.py` |
| **Full analysis with ROC/DET/Bayes plots** | `6_comprehensive_evaluator.py` |
| **Compare DET curves for multiple models** | `7_det_curve_comparison.py` |
| **Detailed DET comparison with EER metrics** | `8_det_curve_zoomed.py` |
| **Analyze audio file characteristics** | `9_audio_dataset_analyzer.py` |
| **Measure dataset similarity/divergence** | `10_dataset_divergence.py` |
| **Visualize clips in 2D (t-SNE/UMAP)** | `clip_visualization.py` |
| **Organize scattered output PNG files** | `5_graph_file_organizer.py` |

---

## Category A: Model Evaluation Scripts

*For analyzing model prediction CSV files (drone_probability, true_label, etc.)*

| Script | Purpose | Usage |
|--------|---------|-------|
| `1_basic_histogram.py` | Quick log-scale histogram visualization | `python 1_basic_histogram.py [dir]` |
| `2_batch_histogram_validator.py` | Recursive CSV processing with column validation, saves PNG | `python 2_batch_histogram_validator.py [dir]` |
| `3_multi_dataset_comparison.py` | Grid comparison with Accuracy/F1/Precision/Recall metrics | `python 3_multi_dataset_comparison.py [dir]` |
| `4_threshold_evaluator.py` | Find optimal decision threshold using F1 scoring | `python 4_threshold_evaluator.py [dir]` |
| `6_comprehensive_evaluator.py` | **Recommended.** Interactive menu: hist/roc/det/bayes/cm plots. Used for thesis figures. | `python 6_comprehensive_evaluator.py [dir]` |
| `7_det_curve_comparison.py` | Multi-model DET curves on same axes (probit scale) | `python 7_det_curve_comparison.py` |
| `8_det_curve_zoomed.py` | DET comparison with zoom inset, EER/AUC metrics table | `python 8_det_curve_zoomed.py` |

**Required CSV columns:** `file_id`, `true_label`, `predicted_label`, `drone_probability`, `aggregation_method`, `aggregation_threshold`, `split`

---

## Category B: Audio Dataset Analysis Scripts

*For analyzing raw audio file properties across datasets*

| Script | Purpose | Input/Output |
|--------|---------|--------------|
| `9_audio_dataset_analyzer.py` | Analyze duration, RMS, peak, crest factor, spectrum across datasets | Configure `COMPARISON_DATASETS`, outputs to `audio_analysis_results/` |
| `10_dataset_divergence.py` | Jensen-Shannon divergence heatmaps, MDS visualization | Requires CSV from `9_audio_dataset_analyzer.py` |
| `clip_visualization.py` | t-SNE/UMAP 2D visualization of audio clips by class/dataset | Requires CSV from `9_audio_dataset_analyzer.py`, outputs to `clip_visualizations/` |
| `divergences2.py` | Simpler variability analysis (Std Dev, IQR, CV bar charts) | - |

---

## Category C: Utility Scripts

| Script | Purpose |
|--------|---------|
| `5_graph_file_organizer.py` | Find and organize `*_histogram.png` files with meaningful names |

---

## Category D: ModelDatasetEpochAnalysis Scripts

*Located in `ModelDatasetEpochAnalysis/` - For cross-model/epoch comparison*

| Script | Purpose |
|--------|---------|
| `1_create_fusion_dataset.py` | **Step 1:** Combine all `*_detailed_files.csv` into Fusion datasets |
| `2_model_ranker.py` | **Step 2:** Rank models by Brier Score/F1, compare calibrated vs optimized thresholds |
| `3_epoch_statistics.py` | Generate per-epoch stats + LaTeX tables (needs rework) |
| `4_dataset_epoch_weighted.py` | Density-weighted probability distribution visualization |
| `5_transformer_weighted.py` | Transformer-specific analysis with KDE density curves |
| `6_basic_histogram.py` | Quick histogram generation |
| `7_organize_graphs.py` | Organize generated PNG files |
| `utils/audio_calculator.py` | Audio file duration/stats utility |

---

## Recommended Workflows

### Model Evaluation

```
1. Run model evaluation → Get prediction CSV files
2. python 6_comprehensive_evaluator.py [dir]  → hist/roc/det/bayes/cm plots
3. python 3_multi_dataset_comparison.py [dir] → Side-by-side comparison
4. python 4_threshold_evaluator.py [dir]      → Optimize threshold
5. python 5_graph_file_organizer.py [dir]     → Organize outputs
```

### Audio Dataset Analysis

```
1. python 9_audio_dataset_analyzer.py         → Stats CSV, histograms, boxplots
2. python 10_dataset_divergence.py            → JS divergence heatmaps
3. python clip_visualization.py (optional)    → t-SNE/UMAP visualizations
```

### ModelDatasetEpochAnalysis

```
1. python 1_create_fusion_dataset.py [dir]    → Combine datasets
2. python 2_model_ranker.py [dir]             → Rank models by Brier/F1
3. python 3_epoch_statistics.py [dir]         → Stats + LaTeX (optional)
4. python 4_dataset_epoch_weighted.py [dir]   → Distribution plots (optional)
5. python 7_organize_graphs.py [dir]          → Organize PNG files
```
