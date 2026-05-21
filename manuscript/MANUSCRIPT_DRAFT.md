# Quantum-Inspired Hybrid AI for Robust RF Signal Classification in Contested Spectrum Environments

## Abstract

Automatic modulation classification is important for spectrum monitoring,
secure wireless communications, cognitive radio, unmanned-system links, and
defense-adjacent RF surveillance. However, many AI-based RF classifiers are
reported mainly under nominal channel conditions and are not evaluated under
contested-spectrum degradations such as low signal-to-noise ratio, jamming,
multipath, carrier-frequency offset, and impulsive interference. This study
develops a reproducible robustness-aware RF signal-classification workflow that
compares classical machine-learning baselines, a compact raw-IQ convolutional
neural network, and a simulated quantum feature-map kernel classifier. The
workflow was first evaluated on a controlled synthetic IQ dataset and then
validated on the public RadioML2016.10A benchmark containing 220,000 examples,
11 modulation classes, and SNR values from -20 to 18 dB. On the RadioML full
GPU run, the raw-IQ CNN achieved the highest clean accuracy (0.5114) and
macro-F1 (0.5231), followed by Random Forest (0.4846 accuracy), RBF-SVM
(0.4739), Logistic Regression (0.4344), and the simulated quantum feature-map
kernel SVM (0.3091). Robustness testing showed major degradation under
narrowband jamming, frequency offset, low SNR, multipath, and impulsive noise,
demonstrating that clean accuracy alone is insufficient for RF model selection.
The quantum-inspired kernel did not outperform classical or deep-learning
baselines; therefore, the contribution is framed as transparent engineering
benchmarking rather than a quantum-advantage claim. The study provides
manuscript-ready tables, figures, code, and reproducibility instructions for
robust RF classification under degraded spectrum conditions.

Keywords: automatic modulation classification; RF signal classification;
quantum-inspired machine learning; quantum kernel; robust artificial
intelligence; contested spectrum; jamming; low SNR; deep learning; cognitive
radio

## 1. Introduction

Automatic modulation classification (AMC) supports RF spectrum monitoring,
adaptive communications, cognitive radio, and RF surveillance workflows. In
practical environments, the received signal is rarely clean. Channel conditions
may include low SNR, narrowband or broadband interference, carrier-frequency
offset, multipath fading, and impulsive noise. A classifier that is validated
only on nominal examples may therefore produce misleading performance
expectations when deployed in degraded or contested spectrum settings.

Recent progress in deep learning has improved AMC performance on public
benchmarks, but robustness evaluation remains uneven across studies. Quantum
and quantum-inspired machine-learning methods are also being explored as
compact nonlinear feature mappings and kernel-based classifiers. For
engineering use, however, such methods must be compared against classical and
deep-learning baselines under identical stress conditions. This work therefore
asks: how do classical ML, raw-IQ CNN, and simulated quantum feature-map kernel
classifiers compare when evaluated under controlled degraded-spectrum
conditions?

## 2. Contributions

This study contributes:

1. A reproducible RF/IQ signal-classification workflow covering synthetic IQ
   data and the public RadioML2016.10A benchmark.
2. A robustness protocol covering clean, low-SNR, narrowband-jam,
   broadband-jam, frequency-offset, multipath, and impulsive-noise conditions.
3. A transparent comparison of engineered-feature classical ML, raw-IQ CNN, and
   simulated quantum feature-map kernel baselines.
4. Full GPU benchmark results on RadioML2016.10A with clean performance,
   stress-condition metrics, robustness-drop tables, and publication-quality
   figures.
5. A cautious interpretation of the quantum-inspired component that avoids
   unsupported quantum-advantage claims.

## 3. Materials And Methods

### 3.1 Datasets

The controlled synthetic pilot uses eight modulation families: BPSK, QPSK,
8PSK, QAM16, QAM64, BFSK, AM-DSB, and FM. Each example contains 128 IQ samples
represented as two channels: in-phase and quadrature. For the scaled pilot, 500
samples per class were generated, giving 4,000 total examples. The synthetic
generator applies randomized phase rotation, carrier offset, optional
multipath, and additive white Gaussian noise across SNR values of -6, 0, 6, 12,
and 18 dB.

The public benchmark validation uses RadioML2016.10A. The converted clean NPZ
contains 220,000 examples, 11 modulation classes, and SNR values from -20 to
18 dB in 2 dB steps. The classes are 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK,
PAM4, QAM16, QAM64, QPSK, and WBFM. The raw RadioML file is not redistributed
with the repository; users must obtain it from the original dataset provider
or validated public mirror and comply with the applicable license terms.

### 3.2 Stress Conditions

Stress-test versions of each held-out example were generated for clean,
low-SNR, narrowband-jamming, broadband-jamming, frequency-offset, multipath,
and impulsive-noise conditions. The purpose of these stressors is not to model
every possible RF environment, but to create a repeatable degradation protocol
that exposes model-specific failure modes.

### 3.3 Models

Classical baselines use engineered RF features derived from IQ statistics,
amplitude statistics, phase statistics, spectral centroid, spectral bandwidth,
and FFT peak-to-mean ratio. The evaluated classical models are Logistic
Regression, Random Forest, and RBF-SVM.

The deep-learning baseline is a compact one-dimensional convolutional neural
network trained directly on raw IQ samples. It uses three Conv1D blocks with
batch normalization, ReLU activations, pooling, adaptive average pooling,
dropout, and a linear classification head. In the full RadioML run, the CNN was
trained on CUDA for 40 epochs with batch size 512 and best-epoch restoration.
The final restored model came from epoch 18.

The quantum-inspired baseline is a simulated quantum feature-map kernel SVM.
Engineered RF features are standardized, reduced with PCA to five dimensions,
scaled to angle values, and mapped to a compact complex statevector using
single-qubit angle features plus pairwise phase interactions. The classifier is
an SVM trained on the resulting fidelity-style precomputed kernel. This is a
simulated quantum-inspired feature map, not a hardware quantum-advantage claim.

### 3.4 Evaluation

All models use stratified held-out evaluation. Clean performance is measured
using accuracy and macro F1. Robustness is measured by evaluating the trained
model on held-out examples transformed by each stress condition and reporting
the percentage drop from clean accuracy and macro F1. The full RadioML run used
30,000 training and 10,000 test examples for the classical baselines, 80,000
training and 20,000 test examples for the raw-IQ CNN, and a compact 120
train-per-class / 80 test-per-class configuration for the simulated quantum
feature-map kernel SVM.

## 4. Results

### 4.1 Synthetic Pilot Results

The synthetic pilot confirmed that the complete workflow runs end to end and
that clean accuracy and robustness ranking can differ. In the 500-sample-per
class synthetic run, Random Forest achieved the highest clean held-out accuracy
(0.567), followed by the raw-IQ CNN (0.545), RBF-SVM (0.544), Logistic
Regression (0.543), and the simulated quantum feature-map kernel SVM (0.423).
The CNN had the lowest mean stress-induced accuracy drop in the synthetic
pilot, but it remained vulnerable to frequency offset and narrowband jamming.

### 4.2 RadioML2016.10A Clean Performance

The full GPU RadioML benchmark is the primary public-dataset evidence for this
paper. Table source:

`manuscript_assets/tables/table_radioml2016_full_clean_performance.csv`

On RadioML2016.10A, the raw-IQ CNN achieved the strongest clean performance
with 0.5114 accuracy and 0.5231 macro-F1 on 20,000 held-out examples. Random
Forest achieved 0.4846 accuracy and 0.4963 macro-F1; RBF-SVM achieved 0.4739
accuracy and 0.4887 macro-F1; Logistic Regression achieved 0.4344 accuracy and
0.4345 macro-F1. The simulated quantum feature-map kernel SVM achieved 0.3091
accuracy and 0.2953 macro-F1 on its compact evaluation subset.

### 4.3 RadioML2016.10A Robustness Under Degraded Conditions

Robustness tables are provided in:

`manuscript_assets/tables/table_radioml2016_full_robustness_metrics.csv`

`manuscript_assets/tables/table_radioml2016_full_robustness_drop.csv`

All models degraded under stress conditions. For the raw-IQ CNN, the largest
accuracy drops were observed under narrowband jamming (82.08 percent drop),
frequency offset (81.61 percent), and impulsive noise (80.28 percent). The CNN
was less affected by broadband jamming (33.64 percent drop) and low SNR
(52.02 percent drop) relative to its most severe failure modes. Random Forest
also degraded sharply under narrowband jamming, while RBF-SVM collapsed toward
near-chance performance across several stressors. These results indicate that
model selection based only on clean-set accuracy would miss important
deployment risks.

### 4.4 Quantum-Inspired Kernel Result

The simulated quantum feature-map kernel baseline did not outperform the
classical or CNN baselines. Its clean RadioML accuracy was 0.3091, and stress
conditions reduced performance toward near-chance levels. This result is still
valuable because it provides a transparent negative benchmark and prevents
overclaiming. The evidence supports a conservative engineering conclusion:
quantum-inspired feature maps can be evaluated as compact nonlinear baselines,
but this experiment does not demonstrate quantum advantage.

## 5. Discussion

The full public benchmark supports the central engineering premise: clean
classification accuracy is insufficient for RF model evaluation. A model may
rank well under clean conditions while failing under a particular stressor. The
raw-IQ CNN produced the best clean RadioML result, yet it degraded sharply
under narrowband jamming and carrier-frequency offset. Classical models also
showed stress-specific collapse, especially under jamming and low-SNR variants.

The gap between clean and degraded performance is important for contested
spectrum applications. Robust RF classifiers should be reported with both
nominal and stress-condition metrics, including the specific perturbations that
cause failure. The robustness-drop tables provide a compact way to compare
models under this requirement.

The quantum-inspired component should be interpreted carefully. The result does
not justify claims of quantum superiority. Instead, it demonstrates how a
simulated quantum feature-map kernel can be included transparently in an
engineering benchmark and compared against practical alternatives.

## 6. Limitations

This study is simulation and public-benchmark based. It does not use live RF
captures, hardware-in-the-loop evaluation, over-the-air adversarial testing, or
classified defense data. The RadioML2016.10A dataset is a useful public
benchmark, but it does not represent every operational RF channel or emitter
environment. The stress transformations are controlled and reproducible rather
than exhaustive. The simulated quantum feature-map kernel is evaluated on a
compact subset because kernel methods scale poorly with the number of training
examples.

## 7. Conclusion

This work establishes a reproducible evaluation workflow for robustness-aware
RF signal classification under degraded spectrum conditions. The full
RadioML2016.10A GPU benchmark shows that the raw-IQ CNN has the strongest clean
performance among the tested models, but also reveals large stress-condition
failure modes. Classical models and the simulated quantum-inspired kernel
provide useful comparators, while the quantum-inspired result should be framed
as a transparent baseline rather than a quantum-advantage result. The main
engineering contribution is a reproducible robustness protocol and evidence
package for AI-based RF classification in contested-spectrum environments.

## References

1. T. J. O'Shea, J. Corgan, and T. C. Clancy, "Convolutional Radio Modulation
   Recognition Networks," arXiv:1602.04105, 2016.
2. DeepSig, "RadioML 2016.10A Dataset," DeepSig Datasets,
   https://www.deepsig.ai/datasets.
3. V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala,
   J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced
   feature spaces," Nature, vol. 567, pp. 209-212, 2019,
   doi: 10.1038/s41586-019-0980-2.
4. M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert
   spaces," Physical Review Letters, vol. 122, 040504, 2019,
   doi: 10.1103/PhysRevLett.122.040504.
