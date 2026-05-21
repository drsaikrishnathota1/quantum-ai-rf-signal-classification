# Quantum-Inspired Hybrid AI for Robust RF Signal Classification in Contested Spectrum Environments

## Abstract

Automatic modulation classification is important for spectrum monitoring,
secure wireless communications, cognitive radio, unmanned-system links, and
defense-adjacent RF surveillance. However, many AI-based classifiers are
reported under nominal channel conditions and are not stress-tested against
contested-spectrum degradations such as low signal-to-noise ratio, jamming,
multipath, carrier-frequency offset, and impulsive interference. This study
develops a reproducible RF signal-classification workflow that compares
classical machine-learning baselines, a raw-IQ convolutional neural network,
and a simulated quantum feature-map kernel classifier under controlled
degradation conditions. A synthetic IQ dataset containing eight modulation
families was generated with controlled SNR variation and seven stress
conditions. Models were evaluated using held-out accuracy, macro F1,
robustness drop from clean conditions, confusion matrices, accuracy by SNR, and
runtime-oriented metadata. In the current synthetic 500-sample-per-class pilot,
Random Forest achieved the highest clean held-out accuracy (0.567), followed
by the raw-IQ CNN (0.545), RBF-SVM (0.544), Logistic Regression (0.543), and the
simulated quantum feature-map kernel SVM (0.423). Robustness analysis showed
that the CNN had the lowest mean stress-induced accuracy drop, but it remained
vulnerable to frequency offset and narrowband jamming. The simulated quantum
feature-map kernel did not outperform classical or deep-learning baselines,
supporting a cautious engineering interpretation rather than a quantum-advantage
claim. The study provides a transparent evaluation protocol for robustness-aware
AI in degraded RF environments and defines the next public-benchmark validation
step using RadioML datasets.

Keywords: automatic modulation classification; RF signal classification;
quantum-inspired machine learning; quantum kernel; robust artificial
intelligence; contested spectrum; jamming; low SNR; deep learning; cognitive
radio

## 1. Introduction

Automatic modulation classification (AMC) supports RF spectrum monitoring,
adaptive communications, cognitive radio, and RF surveillance workflows. In
practical environments, the received signal is rarely clean. Channel conditions
may include low SNR, narrowband or broadband interference, carrier-frequency
offset, multipath fading, and impulsive noise. A classifier that is only
validated on nominal examples may therefore produce misleading performance
expectations.

Recent progress in deep learning has improved AMC performance on benchmark
datasets, but robustness evaluation remains uneven across studies. Quantum and
quantum-inspired machine-learning methods are also being explored for compact
feature mappings and kernel-based classification. For engineering use, however,
these methods must be compared against classical and deep-learning baselines
under the same stress conditions. This work therefore asks: how do classical
ML, raw-IQ CNN, and simulated quantum feature-map kernel classifiers compare
when evaluated under controlled degraded-spectrum conditions?

## 2. Contributions

This study contributes:

1. A reproducible synthetic RF/IQ signal-classification pipeline covering eight
   modulation classes and controlled SNR variation.
2. A robustness protocol covering clean, low-SNR, narrowband-jam,
   broadband-jam, frequency-offset, multipath, and impulsive-noise conditions.
3. A transparent comparison of engineered-feature classical ML, raw-IQ CNN, and
   simulated quantum feature-map kernel baselines.
4. Manuscript-ready tables and figures reporting clean performance, robustness
   drops, best model by stress condition, and average robustness summaries.
5. A cautious interpretation of the quantum-inspired component that avoids
   unsupported quantum-advantage claims.

## 3. Methods

### 3.1 Synthetic IQ Dataset

The current pilot uses a synthetic IQ dataset with eight modulation families:
BPSK, QPSK, 8PSK, QAM16, QAM64, BFSK, AM-DSB, and FM. Each example contains 128
IQ samples represented as two channels: in-phase and quadrature. For the scaled
pilot, 500 samples per class were generated, giving 4,000 total examples.

The generator applies randomized phase rotation, carrier offset, optional
multipath, and additive white Gaussian noise across SNR values of -6, 0, 6, 12,
and 18 dB. Stress-test versions of each example are generated for clean,
low-SNR, narrowband-jamming, broadband-jamming, frequency-offset, multipath,
and impulsive-noise conditions.

### 3.2 Models

Classical baselines use engineered RF features derived from IQ statistics,
amplitude statistics, phase statistics, spectral centroid, spectral bandwidth,
and FFT peak-to-mean ratio. The evaluated classical models are Logistic
Regression, Random Forest, and RBF-SVM.

The deep-learning baseline is a compact one-dimensional convolutional neural
network trained directly on raw IQ samples. It uses three Conv1D blocks with
batch normalization, ReLU activations, pooling, adaptive average pooling,
dropout, and a linear classification head.

The quantum-inspired baseline is a simulated quantum feature-map kernel SVM.
Engineered RF features are standardized, reduced with PCA to five dimensions,
scaled to angle values, and mapped to a compact complex statevector using
single-qubit angle features plus pairwise phase interactions. The classifier is
an SVM trained on the resulting fidelity-style precomputed kernel. This is a
simulated quantum-inspired feature map, not a hardware quantum-advantage claim.

### 3.3 Evaluation

All models use a stratified held-out split. Clean performance is measured using
accuracy and macro F1. Robustness is measured by evaluating the trained model on
held-out examples transformed by each stress condition and reporting the
percentage drop from clean accuracy and macro F1.

## 4. Current Results

### 4.1 Clean Held-Out Performance

Current manuscript table source:

`manuscript_assets/tables/table_synthetic_clean_performance.csv`

Random Forest achieved the strongest clean accuracy in the current synthetic
pilot (0.567), while the raw-IQ CNN (0.545), RBF-SVM (0.544), and Logistic
Regression (0.543) were closely grouped. The simulated quantum feature-map
kernel SVM achieved lower clean accuracy (0.423).

### 4.2 Robustness Under Degraded Conditions

Current manuscript table source:

`manuscript_assets/tables/table_synthetic_robustness_drop.csv`

The largest degradation was observed for the simulated quantum feature-map
kernel under low SNR, Random Forest under narrowband jamming, and the raw-IQ CNN
under frequency offset. The CNN achieved the lowest mean stress-induced
accuracy drop, but this average hides a clear failure mode under frequency
offset.

### 4.3 Model Robustness Summary

Current manuscript table source:

`manuscript_assets/tables/table_synthetic_model_robustness_summary.csv`

The raw-IQ CNN had the best mean stress accuracy in the current pilot, followed
by Logistic Regression and the classical kernel baselines. The quantum-inspired
kernel should be retained as an evaluated baseline, but not presented as the
best-performing model.

## 5. Discussion

The current results support the central engineering premise: clean
classification accuracy is insufficient for RF model evaluation. A model may
rank well under clean conditions while failing under a particular stressor. For
example, the raw-IQ CNN is competitive overall and relatively stable under
multipath and impulsive noise, yet it is highly vulnerable to carrier-frequency
offset. Conversely, Logistic Regression performs better than the CNN under
frequency offset despite having slightly lower clean accuracy.

The quantum-inspired kernel result is scientifically useful even though it does
not outperform the classical and CNN baselines. It prevents overclaiming and
helps frame the contribution as a reproducible robustness comparison rather
than a speculative quantum-performance claim.

## 6. Limitations And Next Validation Step

The current evidence is a synthetic pilot. It is valuable for reproducible
controlled analysis, but it is not sufficient by itself for a strong journal
submission. The next step is to validate the same protocol on a public benchmark
dataset, starting with RadioML2016.10A. The final manuscript should clearly
separate synthetic evidence from public-benchmark evidence and should avoid
redistributing datasets governed by external license terms.

## 7. Conclusion

This work establishes a reproducible evaluation workflow for robustness-aware
RF signal classification under degraded spectrum conditions. The current pilot
shows that robustness ranking differs from clean-accuracy ranking and that
model-specific failure modes should be reported explicitly. The results support
a cautious engineering contribution for Results in Engineering after public
benchmark validation is added.
