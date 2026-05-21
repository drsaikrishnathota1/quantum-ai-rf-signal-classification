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

