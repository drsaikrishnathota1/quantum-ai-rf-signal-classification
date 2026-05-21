# RadioML2016.10A Public Benchmark Smoke Report

This is a fast smoke test proving the public-benchmark workflow works end to end.
It is not the final RadioML result table for the manuscript.

## Dataset

- Source file: `data/radioml/RML2016.10a_dict.pkl`
- Smoke subset: 100 examples per modulation-SNR pair
- Total smoke examples: 22,000
- Modulation classes: 11
- SNR values: -20 to 18 dB in 2 dB steps

## Clean Performance

| model | model_label | examples | accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| rbf_svm | RBF-SVM | 2200.0000 | 0.4555 | 0.4700 |
| random_forest | Random Forest | 2200.0000 | 0.4523 | 0.4603 |
| logistic_regression | Logistic Regression | 2200.0000 | 0.4300 | 0.4293 |
| iq_cnn | Raw-IQ CNN | 2200.0000 | 0.3768 | 0.3549 |
| simulated_quantum_feature_kernel_svm | Simulated QFM-Kernel SVM | 330.0000 | 0.3000 | 0.2850 |

## Worst Robustness Drops

| model_label | condition | accuracy | accuracy_drop_pct | macro_f1_drop_pct |
| --- | --- | --- | --- | --- |
| RBF-SVM | frequency_offset | 0.0909 | 80.0399 | 96.7763 |
| RBF-SVM | low_snr | 0.0909 | 80.0399 | 96.7763 |
| RBF-SVM | impulsive_noise | 0.0909 | 80.0399 | 96.7763 |
| RBF-SVM | multipath | 0.0909 | 80.0399 | 96.7763 |
| RBF-SVM | broadband_jam | 0.0909 | 80.0399 | 96.7763 |
| RBF-SVM | narrowband_jam | 0.0909 | 80.0399 | 96.7763 |
| Random Forest | narrowband_jam | 0.0909 | 79.8995 | 96.7082 |
| Logistic Regression | impulsive_noise | 0.0891 | 79.2812 | 95.6337 |

## Interpretation

The RadioML smoke run confirms that the public-benchmark ingestion, stress generation, classical baselines, CNN baseline, and simulated quantum feature-map kernel baseline all execute successfully. The current CNN result uses a local CPU smoke run with best-epoch restoration from a 30-epoch training history. Final manuscript claims should use a longer GPU run on the full RadioML dataset and report RadioML results separately from synthetic results.
