# Submission Package Checklist

Target journal: **Results in Engineering**  
Article type: **Original Article**

## Current Readiness

Status: **not ready for final submission yet, but public-benchmark ingestion is now validated**

Reason:

- The synthetic experiment pipeline is working.
- Classical, CNN, and simulated quantum-kernel baselines are implemented.
- RadioML2016.10A smoke validation now runs end to end.
- A Results in Engineering submission should still include a longer GPU-based
  public-benchmark run before submission.

## Files To Prepare For Final Upload

Required:

- Cover Letter
- Abstract
- Title Page with author details
- Manuscript without author details
- Conflict of Interest statement
- Highlights
- Figures at publication quality
- Tables, either embedded in manuscript or uploaded separately

Recommended:

- Graphical abstract
- Supplementary code package or repository link
- Data and code availability statement
- Reproducibility checklist
- Response-to-reviewers file only if this is a revision

## Repository Files Already Available

- `submission_files/MANUAL_SUBMISSION_SCRIPT.md`
- `manuscript/MANUSCRIPT_DRAFT.md`
- `manuscript/TITLE_PAGE.md`
- `manuscript/COVER_LETTER_DRAFT.md`
- `manuscript/HIGHLIGHTS.md`
- `manuscript/DECLARATIONS.md`
- `ANALYSIS_REPORT_SYNTHETIC_500.md`
- `manuscript_assets/tables/table_synthetic_clean_performance.csv`
- `manuscript_assets/tables/table_synthetic_robustness_metrics.csv`
- `manuscript_assets/tables/table_synthetic_robustness_drop.csv`
- `manuscript_assets/tables/table_synthetic_model_robustness_summary.csv`
- `manuscript_assets/tables/table_synthetic_best_model_by_condition.csv`
- `manuscript_assets/figures/fig_synthetic_clean_accuracy.png`
- `manuscript_assets/figures/fig_synthetic_robustness_drop_heatmap.png`

## Must Complete Before Submission

1. Add RadioML2016.10A validation.
2. Add at least one stronger deep-learning model or ablation, if runtime allows.
3. Add final accuracy-by-SNR figure across all main models.
4. Add final confusion matrices for clean and worst stress condition.
5. Add final limitations section explaining synthetic and public-data scope.
6. Convert the final manuscript into DOCX.
7. Check journal formatting, word count, figure quality, and references.
