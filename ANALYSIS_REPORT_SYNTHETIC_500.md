# Synthetic 500-Sample RF Analysis Report

This report aggregates the corrected held-out synthetic RF evaluation.

## Clean Performance

| model | model_label | accuracy | macro_f1 |
| --- | --- | --- | --- |
| random_forest | Random Forest | 0.5670 | 0.5650 |
| rbf_svm | RBF-SVM | 0.5440 | 0.5459 |
| logistic_regression | Logistic Regression | 0.5430 | 0.5414 |
| simulated_quantum_feature_kernel_svm | Simulated QFM-Kernel SVM | 0.4229 | 0.4224 |

## Worst Robustness Drops

| model_label | condition | accuracy | accuracy_drop_pct | macro_f1_drop_pct |
| --- | --- | --- | --- | --- |
| Simulated QFM-Kernel SVM | low_snr | 0.1562 | 63.0542 | 65.7142 |
| Random Forest | narrowband_jam | 0.2140 | 62.2575 | 78.5537 |
| RBF-SVM | narrowband_jam | 0.2380 | 56.2500 | 68.0827 |
| Logistic Regression | impulsive_noise | 0.2530 | 53.4070 | 60.8237 |
| Random Forest | impulsive_noise | 0.2710 | 52.2046 | 61.3722 |
| RBF-SVM | low_snr | 0.2620 | 51.8382 | 49.5123 |
| Logistic Regression | narrowband_jam | 0.2630 | 51.5654 | 63.9772 |
| Simulated QFM-Kernel SVM | impulsive_noise | 0.2083 | 50.7389 | 63.4820 |

## Interpretation

Random Forest provided the strongest clean accuracy among the classical pilot models, but all models showed substantial degradation under low-SNR, narrowband jamming, broadband jamming, and impulsive-noise conditions. The simulated quantum feature-map kernel baseline is currently weaker than the best classical model, which is important because the manuscript should not claim quantum advantage. Instead, the result supports a careful engineering framing: quantum-inspired feature maps can be evaluated as compact robustness-aware modules, but they must be compared against strong classical baselines.
