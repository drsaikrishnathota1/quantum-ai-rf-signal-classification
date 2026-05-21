# RadioML2016.10A Full GPU Benchmark Report

This report aggregates the full public-benchmark workflow executed on RadioML2016.10A.
It should be used as the primary manuscript evidence for the public RF benchmark.

## Dataset

- Dataset: RadioML2016.10A
- Total examples in converted clean NPZ: 220,000
- Modulation classes: 11
- SNR values: -20 to 18 dB in 2 dB steps
- Stress conditions evaluated: clean, low_snr, narrowband_jam, broadband_jam, frequency_offset, multipath, impulsive_noise

## Execution Notes

- Classical models used capped stratified training and test subsets for tractable baseline comparison.
- Raw-IQ CNN used CUDA GPU training with best-epoch restoration.
- The quantum-inspired result is a simulated feature-map kernel baseline; no quantum advantage is claimed.

## Method Configuration

| method | models | train_examples | test_examples |
| --- | --- | --- | --- |
| Classical baselines | Logistic Regression; Random Forest; RBF-SVM | 30000 | 10000 |
| Raw-IQ CNN | Compact 1D convolutional neural network | 80000 | 20000 |
| Simulated QFM-Kernel SVM | Five-qubit simulated feature-map kernel with SVM | 120 per class cap | 80 per class cap |

## Clean Performance

| model | model_label | examples | accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| iq_cnn | Raw-IQ CNN | 20000 | 0.5114 | 0.5231 |
| random_forest | Random Forest | 10000 | 0.4846 | 0.4963 |
| rbf_svm | RBF-SVM | 10000 | 0.4739 | 0.4887 |
| logistic_regression | Logistic Regression | 10000 | 0.4344 | 0.4345 |
| simulated_quantum_feature_kernel_svm | Simulated QFM-Kernel SVM | 880 | 0.3091 | 0.2953 |

## Best Clean Result

- Best clean model: Raw-IQ CNN
- Clean accuracy: 0.5114
- Clean macro-F1: 0.5231

## Worst Robustness Drops

| model_label | condition | accuracy | accuracy_drop_pct | macro_f1_drop_pct |
| --- | --- | --- | --- | --- |
| Raw-IQ CNN | narrowband_jam | 0.0916 | 82.0804 | 91.9298 |
| Raw-IQ CNN | frequency_offset | 0.0940 | 81.6111 | 93.0547 |
| Random Forest | narrowband_jam | 0.0909 | 81.2423 | 96.9474 |
| RBF-SVM | broadband_jam | 0.0909 | 80.8187 | 96.9002 |
| RBF-SVM | narrowband_jam | 0.0909 | 80.8187 | 96.9002 |
| RBF-SVM | multipath | 0.0909 | 80.8187 | 96.9002 |
| RBF-SVM | impulsive_noise | 0.0909 | 80.8187 | 96.9002 |
| RBF-SVM | frequency_offset | 0.0909 | 80.8187 | 96.9002 |
| RBF-SVM | low_snr | 0.0909 | 80.8187 | 96.9002 |
| Raw-IQ CNN | impulsive_noise | 0.1008 | 80.2816 | 89.1102 |

## Generated Tables

- `manuscript_assets/tables/table_radioml2016_full_clean_performance.csv`
- `manuscript_assets/tables/table_radioml2016_full_robustness_metrics.csv`
- `manuscript_assets/tables/table_radioml2016_full_robustness_drop.csv`
- `manuscript_assets/tables/table_radioml2016_full_method_config.csv`

## Generated Figures

- `manuscript_assets/figures/fig_radioml2016_clean_accuracy.png`
- `manuscript_assets/figures/fig_radioml2016_robustness_drop.png`

## Manuscript Interpretation

The full public benchmark supports a cautious engineering conclusion: the compact Raw-IQ CNN provided the strongest clean-set performance among the tested methods, while stress tests show large degradation under contested-spectrum perturbations. The simulated quantum feature-map kernel is useful as a transparent quantum-inspired comparator, but the observed performance does not support a quantum-advantage claim.
