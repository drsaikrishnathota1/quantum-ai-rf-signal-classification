# Synthetic 500-Sample RF Analysis Report

This report aggregates the corrected held-out synthetic RF evaluation.

## Clean Performance

| model | model_label | accuracy | macro_f1 |
| --- | --- | --- | --- |
| random_forest | Random Forest | 0.5670 | 0.5650 |
| iq_cnn | Raw-IQ CNN | 0.5450 | 0.4633 |
| rbf_svm | RBF-SVM | 0.5440 | 0.5459 |
| logistic_regression | Logistic Regression | 0.5430 | 0.5414 |
| simulated_quantum_feature_kernel_svm | Simulated QFM-Kernel SVM | 0.4229 | 0.4224 |

## Accuracy By SNR

| model_label | snr_db | accuracy | macro_f1 |
| --- | --- | --- | --- |
| Logistic Regression | -6.0000 | 0.3350 | 0.3308 |
| Logistic Regression | 0.0000 | 0.4500 | 0.4381 |
| Logistic Regression | 6.0000 | 0.5865 | 0.5572 |
| Logistic Regression | 12.0000 | 0.6601 | 0.6810 |
| Logistic Regression | 18.0000 | 0.6823 | 0.6520 |
| Random Forest | -6.0000 | 0.2995 | 0.2888 |
| Random Forest | 0.0000 | 0.4150 | 0.4233 |
| Random Forest | 6.0000 | 0.6298 | 0.6056 |
| Random Forest | 12.0000 | 0.7094 | 0.7278 |
| Random Forest | 18.0000 | 0.7812 | 0.7602 |
| RBF-SVM | -6.0000 | 0.2893 | 0.2907 |
| RBF-SVM | 0.0000 | 0.4300 | 0.4275 |
| RBF-SVM | 6.0000 | 0.5769 | 0.5662 |
| RBF-SVM | 12.0000 | 0.7291 | 0.7402 |
| RBF-SVM | 18.0000 | 0.6927 | 0.6741 |
| Raw-IQ CNN | -6.0000 | 0.4010 | 0.3367 |
| Raw-IQ CNN | 0.0000 | 0.4750 | 0.4156 |
| Raw-IQ CNN | 6.0000 | 0.6442 | 0.5413 |
| Raw-IQ CNN | 12.0000 | 0.5862 | 0.5166 |
| Raw-IQ CNN | 18.0000 | 0.6146 | 0.5445 |

## Worst Robustness Drops

| model_label | condition | accuracy | accuracy_drop_pct | macro_f1_drop_pct |
| --- | --- | --- | --- | --- |
| Simulated QFM-Kernel SVM | low_snr | 0.1562 | 63.0542 | 65.7142 |
| Random Forest | narrowband_jam | 0.2140 | 62.2575 | 78.5537 |
| Raw-IQ CNN | frequency_offset | 0.2300 | 57.7982 | 70.0413 |
| RBF-SVM | narrowband_jam | 0.2380 | 56.2500 | 68.0827 |
| Raw-IQ CNN | narrowband_jam | 0.2510 | 53.9450 | 54.6430 |
| Logistic Regression | impulsive_noise | 0.2530 | 53.4070 | 60.8237 |
| Random Forest | impulsive_noise | 0.2710 | 52.2046 | 61.3722 |
| RBF-SVM | low_snr | 0.2620 | 51.8382 | 49.5123 |

## Model Robustness Summary

| model_label | mean_stress_accuracy | mean_accuracy_drop_pct | worst_accuracy_drop_pct | worst_condition |
| --- | --- | --- | --- | --- |
| Raw-IQ CNN | 0.3750 | 31.1927 | 57.7982 | frequency_offset |
| Logistic Regression | 0.3355 | 38.2136 | 53.4070 | impulsive_noise |
| Simulated QFM-Kernel SVM | 0.2556 | 39.5731 | 63.0542 | low_snr |
| RBF-SVM | 0.3270 | 39.8897 | 56.2500 | narrowband_jam |
| Random Forest | 0.3270 | 42.3280 | 62.2575 | narrowband_jam |

## Best Model By Stress Condition

| condition | model_label | accuracy | macro_f1 |
| --- | --- | --- | --- |
| broadband_jam | Raw-IQ CNN | 0.4380 | 0.3725 |
| frequency_offset | Logistic Regression | 0.4710 | 0.4787 |
| impulsive_noise | Raw-IQ CNN | 0.4430 | 0.4156 |
| low_snr | Raw-IQ CNN | 0.3880 | 0.3237 |
| multipath | Raw-IQ CNN | 0.5000 | 0.4646 |
| narrowband_jam | Logistic Regression | 0.2630 | 0.1950 |

## Interpretation

Random Forest provided the strongest clean accuracy among the classical pilot models. The raw-IQ CNN had competitive clean accuracy and was comparatively stable under multipath and impulsive-noise stress, but it was vulnerable to frequency offset and narrowband jamming. The simulated quantum feature-map kernel baseline is currently weaker than the best classical and CNN baselines, which is important because the manuscript should not claim quantum advantage. Instead, the result supports a careful engineering framing: quantum-inspired feature maps can be evaluated as compact robustness-aware modules, but they must be compared transparently against strong classical and deep-learning baselines.
