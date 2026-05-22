# Synthetic 500-Sample RF Analysis Report

This report aggregates the corrected held-out synthetic RF evaluation.

## Clean Performance

| model | model_label | accuracy | macro_f1 |
| --- | --- | --- | --- |
| iq_cnn | Raw-IQ CNN | 0.6535 | 0.6232 |
| random_forest | Random Forest | 0.5923 | 0.5897 |
| rbf_svm | RBF-SVM | 0.5697 | 0.5659 |
| logistic_regression | Logistic Regression | 0.5543 | 0.5433 |
| simulated_quantum_feature_kernel_svm | Simulated QFM-Kernel SVM | 0.4104 | 0.4080 |
| classical_pca_rbf_svm | Classical PCA-RBF SVM | 0.4083 | 0.4054 |

## Accuracy By SNR

| model_label | snr_db | accuracy | macro_f1 |
| --- | --- | --- | --- |
| Logistic Regression | -6.0000 | 0.3001 | 0.2906 |
| Logistic Regression | 0.0000 | 0.4823 | 0.4671 |
| Logistic Regression | 6.0000 | 0.6042 | 0.5800 |
| Logistic Regression | 12.0000 | 0.6650 | 0.6500 |
| Logistic Regression | 18.0000 | 0.7127 | 0.7098 |
| Random Forest | -6.0000 | 0.3474 | 0.3397 |
| Random Forest | 0.0000 | 0.4901 | 0.4822 |
| Random Forest | 6.0000 | 0.6683 | 0.6597 |
| Random Forest | 12.0000 | 0.6699 | 0.6641 |
| Random Forest | 18.0000 | 0.7774 | 0.7906 |
| RBF-SVM | -6.0000 | 0.2927 | 0.2866 |
| RBF-SVM | 0.0000 | 0.4757 | 0.4648 |
| RBF-SVM | 6.0000 | 0.6289 | 0.6142 |
| RBF-SVM | 12.0000 | 0.6626 | 0.6546 |
| RBF-SVM | 18.0000 | 0.7811 | 0.7878 |
| Raw-IQ CNN | -6.0000 | 0.4184 | 0.3965 |
| Raw-IQ CNN | 0.0000 | 0.6321 | 0.5850 |
| Raw-IQ CNN | 6.0000 | 0.7201 | 0.6931 |
| Raw-IQ CNN | 12.0000 | 0.7613 | 0.7271 |
| Raw-IQ CNN | 18.0000 | 0.7313 | 0.7160 |

## Worst Robustness Drops

| model_label | condition | accuracy | accuracy_drop_pct | macro_f1_drop_pct |
| --- | --- | --- | --- | --- |
| Raw-IQ CNN | narrowband_jam | 0.2045 | 68.7070 | 76.1164 |
| Raw-IQ CNN | impulsive_noise | 0.2180 | 66.6412 | 69.9001 |
| RBF-SVM | narrowband_jam | 0.2030 | 64.3703 | 78.1879 |
| Random Forest | narrowband_jam | 0.2150 | 63.6978 | 83.2318 |
| Raw-IQ CNN | frequency_offset | 0.2460 | 62.3565 | 71.2983 |
| Simulated QFM-Kernel SVM | impulsive_noise | 0.1604 | 60.9137 | 70.3632 |
| Classical PCA-RBF SVM | low_snr | 0.1646 | 59.6939 | 65.8119 |
| Logistic Regression | narrowband_jam | 0.2435 | 56.0668 | 68.8280 |

## Model Robustness Summary

| model_label | mean_stress_accuracy | mean_accuracy_drop_pct | worst_accuracy_drop_pct | worst_condition |
| --- | --- | --- | --- | --- |
| Classical PCA-RBF SVM | 0.2531 | 38.0102 | 59.6939 | low_snr |
| Logistic Regression | 0.3376 | 39.0919 | 56.0668 | narrowband_jam |
| Simulated QFM-Kernel SVM | 0.2455 | 40.1861 | 60.9137 | impulsive_noise |
| RBF-SVM | 0.3228 | 43.3450 | 64.3703 | narrowband_jam |
| Random Forest | 0.3263 | 44.8994 | 63.6978 | narrowband_jam |
| Raw-IQ CNN | 0.3400 | 47.9725 | 68.7070 | narrowband_jam |

## Best Model By Stress Condition

| condition | model_label | accuracy | macro_f1 |
| --- | --- | --- | --- |
| broadband_jam | Raw-IQ CNN | 0.5075 | 0.4833 |
| frequency_offset | Logistic Regression | 0.4760 | 0.4644 |
| impulsive_noise | Random Forest | 0.2670 | 0.2117 |
| low_snr | Raw-IQ CNN | 0.4143 | 0.3971 |
| multipath | Raw-IQ CNN | 0.4497 | 0.4480 |
| narrowband_jam | Logistic Regression | 0.2435 | 0.1694 |

## Interpretation

Random Forest provided the strongest clean accuracy among the classical pilot models. The raw-IQ CNN had competitive clean accuracy and was comparatively stable under multipath and impulsive-noise stress, but it was vulnerable to frequency offset and narrowband jamming. The simulated quantum feature-map kernel baseline is currently weaker than the best classical and CNN baselines, which is important because the manuscript should not claim quantum advantage. Instead, the result supports a careful engineering framing: quantum-inspired feature maps can be evaluated as compact robustness-aware modules, but they must be compared transparently against strong classical and deep-learning baselines.
