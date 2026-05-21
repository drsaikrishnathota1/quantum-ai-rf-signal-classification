# Experiment Results Log

## 2026-05-21 Local Synthetic Pilot

Purpose:

Validate the full RF classification workflow before using paid GPU compute.

Command:

```bash
.venv/bin/python scripts/generate_synthetic_iq_dataset.py --samples-per-class 60 --out data/pilot_iq_small.npz
.venv/bin/python scripts/train_pilot_classifiers.py --data data/pilot_iq_small.npz --out results/pilot_small
.venv/bin/python scripts/make_pilot_figures.py --metrics results/pilot_small/pilot_metrics.csv --drops results/pilot_small/pilot_robustness_drop.csv --out results/pilot_small/figures
```

Dataset:

- Synthetic IQ signals
- 8 classes: BPSK, QPSK, 8PSK, QAM16, QAM64, BFSK, AM-DSB, FM
- 60 samples per class
- 480 total examples
- IQ length: 128

Stress conditions:

- clean
- low_snr
- narrowband_jam
- broadband_jam
- frequency_offset
- multipath
- impulsive_noise

Clean-condition pilot accuracy:

| Model | Clean Accuracy | Clean Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.5833 | 0.5757 |
| Random Forest | 0.8583 | 0.8590 |
| RBF-SVM | 0.7125 | 0.7120 |

Key pilot observation:

Even with a small local synthetic dataset, stress conditions reduce classifier
performance sharply. Low SNR, narrowband jamming, broadband jamming, and
impulsive noise are already visible as major risk factors. This supports the
paper's central engineering framing: RF classifiers should be evaluated under
contested-spectrum degradation, not only clean/nominal data.

Generated artifacts:

- `results/pilot_small/pilot_metrics.csv`
- `results/pilot_small/pilot_robustness_drop.csv`
- `results/pilot_small/summary.json`
- `results/pilot_small/figures/pilot_clean_accuracy.png`
- `results/pilot_small/figures/pilot_robustness_drop.png`

Important limitation:

This is only a sanity-test pilot. It is not paper-grade evidence yet. The next
step is to scale the synthetic dataset and add a public RadioML benchmark.

## 2026-05-21 Scaled Synthetic Held-Out Analysis

Purpose:

Run a larger synthetic RF experiment with corrected held-out split logic. The
stress-condition evaluation now uses the same held-out test indices as clean
evaluation, avoiding optimistic full-dataset scoring.

Commands:

```bash
.venv/bin/python scripts/generate_synthetic_iq_dataset.py --samples-per-class 500 --out data/synthetic_iq_500.npz
.venv/bin/python scripts/train_pilot_classifiers.py --data data/synthetic_iq_500.npz --out results/synthetic_500
.venv/bin/python scripts/train_quantum_inspired_kernel.py --data data/synthetic_iq_500.npz --out results/quantum_kernel_500 --qubits 5 --max-train-per-class 90 --max-test-per-class 60
.venv/bin/python scripts/aggregate_synthetic_analysis.py
```

Dataset:

- Synthetic IQ signals
- 8 modulation classes
- 500 samples per class
- 4,000 total examples
- IQ length: 128

Clean held-out performance:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Random Forest | 0.5670 | 0.5650 |
| RBF-SVM | 0.5440 | 0.5459 |
| Logistic Regression | 0.5430 | 0.5414 |
| Simulated QFM-Kernel SVM | 0.4229 | 0.4224 |

Worst observed drops:

| Model | Stress condition | Accuracy | Accuracy drop % |
| --- | --- | ---: | ---: |
| Simulated QFM-Kernel SVM | low_snr | 0.1562 | 63.05 |
| Random Forest | narrowband_jam | 0.2140 | 62.26 |
| RBF-SVM | narrowband_jam | 0.2380 | 56.25 |
| Logistic Regression | impulsive_noise | 0.2530 | 53.41 |

Interpretation:

The scaled held-out pilot supports the paper's central engineering premise:
nominal RF classification accuracy is not enough. Low SNR, narrowband jamming,
and impulsive noise produce large performance degradation across classical and
quantum-inspired baselines. The simulated quantum feature-map kernel is weaker
than the best classical models in this first run, so the paper should avoid
quantum-advantage claims and instead position the quantum component as an
evaluated compact feature-map baseline.

Generated paper assets:

- `ANALYSIS_REPORT_SYNTHETIC_500.md`
- `manuscript_assets/tables/table_synthetic_clean_performance.csv`
- `manuscript_assets/tables/table_synthetic_robustness_metrics.csv`
- `manuscript_assets/tables/table_synthetic_robustness_drop.csv`
- `manuscript_assets/figures/fig_synthetic_clean_accuracy.png`
- `manuscript_assets/figures/fig_synthetic_robustness_drop_heatmap.png`

## 2026-05-21 Raw-IQ CNN Baseline And Updated Synthetic Analysis

Purpose:

Add a deep-learning baseline trained directly on raw IQ samples, then regenerate
the manuscript-ready tables and figures so the current analysis compares
classical ML, CNN, and simulated quantum feature-map kernel models.

Commands:

```bash
.venv/bin/python scripts/train_cnn_iq_baseline.py --data data/synthetic_iq_500.npz --out results/cnn_500 --epochs 18 --batch-size 128
.venv/bin/python scripts/train_pilot_classifiers.py --data data/synthetic_iq_500.npz --out results/synthetic_500
.venv/bin/python scripts/aggregate_synthetic_analysis.py
```

Clean held-out performance after adding CNN:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Random Forest | 0.5670 | 0.5650 |
| Raw-IQ CNN | 0.5450 | 0.4633 |
| RBF-SVM | 0.5440 | 0.5459 |
| Logistic Regression | 0.5430 | 0.5414 |
| Simulated QFM-Kernel SVM | 0.4229 | 0.4224 |

CNN robustness observations:

| Stress condition | Accuracy | Accuracy drop % | Macro F1 |
| --- | ---: | ---: | ---: |
| low_snr | 0.3880 | 28.81 | 0.3237 |
| narrowband_jam | 0.2510 | 53.94 | 0.2101 |
| broadband_jam | 0.4380 | 19.63 | 0.3725 |
| frequency_offset | 0.2300 | 57.80 | 0.1388 |
| multipath | 0.5000 | 8.26 | 0.4646 |
| impulsive_noise | 0.4430 | 18.72 | 0.4156 |

Interpretation:

The raw-IQ CNN is competitive with the best clean classical baselines and has
the lowest mean accuracy drop across stress conditions in the current pilot.
However, it has a strong failure mode under frequency offset and narrowband
jamming. This is useful for the manuscript because it supports a nuanced
robustness argument instead of a simple clean-accuracy ranking.

New generated assets:

- `results/cnn_500/cnn_metrics.csv`
- `results/cnn_500/cnn_robustness_drop.csv`
- `results/cnn_500/cnn_training_history.csv`
- `results/cnn_500/cnn_accuracy_by_snr.csv`
- `results/synthetic_500/accuracy_by_snr.csv`
- `manuscript_assets/tables/table_synthetic_model_robustness_summary.csv`
- `manuscript_assets/tables/table_synthetic_best_model_by_condition.csv`

## 2026-05-21 RadioML2016.10A Public Benchmark Smoke Test

Purpose:

Validate that the public RadioML2016.10A benchmark can be ingested and evaluated
with the same classical, CNN, and simulated quantum-kernel pipeline. This is a
fast proof-of-flow, not the final paper-grade RadioML result.

Input:

- Source file: `data/radioml/RML2016.10a_dict.pkl`
- Smoke subset: 100 examples per modulation-SNR pair
- Total smoke examples: 22,000
- Classes: 11
- SNR values: -20 to 18 dB in 2 dB steps

Commands:

```bash
.venv/bin/python scripts/prepare_radioml2016_npz.py --input data/radioml/RML2016.10a_dict.pkl --out data/radioml/radioml2016_10a_clean_smoke.npz --max-examples-per-mod-snr 100
.venv/bin/python scripts/add_stress_conditions_to_npz.py --input data/radioml/radioml2016_10a_clean_smoke.npz --out data/radioml/radioml2016_10a_stress_smoke.npz
.venv/bin/python scripts/train_pilot_classifiers.py --data data/radioml/radioml2016_10a_stress_smoke.npz --out results/radioml2016_smoke_classical --max-train-examples 6000 --max-test-examples 2200
.venv/bin/python scripts/train_cnn_iq_baseline.py --data data/radioml/radioml2016_10a_stress_smoke.npz --out results/radioml2016_smoke_cnn --epochs 5 --batch-size 256 --max-train-examples 6000 --max-test-examples 2200
.venv/bin/python scripts/train_quantum_inspired_kernel.py --data data/radioml/radioml2016_10a_stress_smoke.npz --out results/radioml2016_smoke_quantum_kernel --qubits 5 --max-train-per-class 50 --max-test-per-class 30
.venv/bin/python scripts/aggregate_radioml_smoke_analysis.py
```

Clean smoke performance:

| Model | Examples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| RBF-SVM | 2200 | 0.4555 | 0.4700 |
| Random Forest | 2200 | 0.4523 | 0.4603 |
| Logistic Regression | 2200 | 0.4300 | 0.4293 |
| Raw-IQ CNN | 2200 | 0.3214 | 0.2698 |
| Simulated QFM-Kernel SVM | 330 | 0.3000 | 0.2850 |

Interpretation:

The RadioML smoke run proves the public-benchmark workflow is connected and
reproducible. The CNN result is intentionally under-trained because it used
only five local CPU epochs. Final RadioML claims should be produced with a
longer GPU run, likely on RunPod, and should be reported separately from the
synthetic controlled-stress results.
