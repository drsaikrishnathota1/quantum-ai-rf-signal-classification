# RSC-Bench: Robustness Benchmarking of Classical, Deep, and Quantum-Inspired AI for RF Signal Classification in Contested Spectrum Environments

## Abstract

Automatic modulation classification is important for spectrum monitoring,
secure wireless communications, cognitive radio, unmanned-system links, and
defense-adjacent RF surveillance. However, many AI-based RF classifiers are
reported mainly under nominal channel assumptions, while practical receivers
must operate under low signal-to-noise ratio, narrowband and broadband
jamming, carrier-frequency offset, multipath, and impulsive interference. This
study develops RSC-Bench, a reproducible robustness-benchmarking protocol that
compares classical machine-learning baselines, a compact raw-IQ convolutional
neural network, a simulated quantum feature-map kernel classifier, and a
classical PCA-RBF SVM ablation baseline. The novelty of this work is
RSC-Bench, a reproducible robustness-evaluation protocol that couples clean RF
modulation classification with stress-condition operators, same-feature
quantum/classical kernel ablation, and complexity reporting on both synthetic
IQ data and RadioML2016.10A. The protocol was evaluated on a controlled
synthetic IQ dataset with 16,000 examples and eight modulation families, and
validated on the public RadioML2016.10A benchmark containing 220,000 examples,
11 modulation classes, and SNR values from -20 to 18 dB. On the synthetic
dataset, the raw-IQ CNN achieved the highest clean accuracy (0.654), followed
by Random Forest (0.592), RBF-SVM (0.570), Logistic Regression (0.554), the
simulated quantum kernel SVM (0.410), and the PCA-RBF ablation (0.408). On
RadioML2016.10A, the raw-IQ CNN achieved the strongest clean accuracy (0.511)
and macro-F1 (0.523). Robustness testing showed severe degradation under
narrowband jamming, carrier-frequency offset, low SNR, multipath, and
impulsive noise, demonstrating that clean accuracy alone is insufficient for RF
model selection. The quantum-inspired kernel did not outperform the classical
PCA-RBF ablation on the same compressed features, confirming that no
quantum-feature-map advantage is supported by the evidence. The main
engineering contribution is a transparent, reproducible robustness protocol
with full public-benchmark validation, stress-condition failure analysis,
ablation evidence, and computational complexity reporting.

Keywords: automatic modulation classification; RF signal classification;
quantum-inspired machine learning; quantum kernel; robust artificial
intelligence; contested spectrum; jamming; low SNR; deep learning; cognitive
radio

## 1. Introduction

Automatic modulation classification (AMC) aims to identify the modulation type
of an unknown received signal without requiring full transmitter-side
metadata. It is a long-standing problem in signal processing and is relevant to
adaptive communications, cognitive radio, interference monitoring, spectrum
management, and RF surveillance. Classical AMC work relied heavily on
likelihood-based methods, higher-order statistics, cumulants, cyclostationary
features, and engineered time/frequency-domain descriptors [1, 2]. More recent
work has shown that deep neural networks can learn useful representations
directly from raw IQ samples or derived signal representations [3-13, 26-34].

Despite this progress, a gap remains between clean benchmark performance and
operational robustness. Practical receivers may observe signals distorted by
low SNR, frequency offset, fading, multipath, impulsive interference, and
intentional jamming. Deep RF classifiers are also known to be vulnerable to
adversarial perturbations and over-the-air attack settings [15-20]. For this
reason, nominal clean accuracy is not sufficient for an engineering assessment
of RF classifiers intended for degraded or contested spectrum environments.

Quantum and quantum-inspired machine-learning methods have also attracted
attention because quantum feature maps can be interpreted as high-dimensional
kernel embeddings [21-25]. However, quantum advantage in practical
classical-data classification problems is not automatic. Recent analyses show
that classical models can remain competitive when enough data are available,
and that quantum-kernel methods must be assessed against strong classical
kernel baselines rather than weaker straw-man comparators [23-25].

This paper therefore does not present a new state-of-the-art AMC architecture
or claim quantum advantage. Instead, it addresses a narrower but important
engineering question: when classical ML, raw-IQ CNN, and simulated
quantum-feature-map kernel classifiers are evaluated under the same clean and
degraded RF conditions, what failure modes appear, and does the
quantum-inspired embedding add measurable value beyond a classical kernel on
the same reduced feature space?

The novelty of this work is RSC-Bench, a reproducible robustness-evaluation
protocol that couples clean RF modulation classification with stress-condition
operators, same-feature quantum/classical kernel ablation, and complexity
reporting on both synthetic IQ data and RadioML2016.10A.

## 2. Related Work And Research Gap

### 2.1 Classical AMC and benchmark evolution

Classical AMC surveys established the two main families of modulation
recognition methods: likelihood-based approaches and feature-based approaches
[1]. Likelihood-based methods can approach optimality under well-specified
models, but they are often computationally expensive and sensitive to
assumptions about channel state, timing, phase, and SNR. Feature-based methods
reduce complexity by using signal statistics, cumulants, instantaneous
amplitude/phase features, and spectral features, but their performance depends
on feature design and channel assumptions [1, 2].

The RadioML family of datasets helped standardize deep-learning evaluation for
radio signal classification [3-5]. RadioML2016.10A remains widely used because
it provides labeled IQ sequences across multiple modulation types and SNR
levels. Later works extended the benchmark setting with deeper architectures,
larger datasets, distributed sensors, multistream fusion,
resource-constrained models, attention mechanisms, transformer blocks, and
joint signal-detection/classification settings [5-13, 26-34].

### 2.2 Deep learning for RF signal classification

Convolutional modulation-recognition networks demonstrated that raw IQ
sequences can be learned directly by CNNs [3]. Deep architectures were then
expanded through residual networks, CLDNN-style models, recurrent models,
feature-fusion systems, attention mechanisms, transformer encoders, and
multi-domain input representations [4-13, 26-34]. These works are essential
because they show that deep models can outperform many hand-crafted feature
pipelines, especially at moderate and high SNR.

However, deep AMC literature often emphasizes clean or SNR-stratified
performance rather than a broader stress-condition matrix. Some papers report
accuracy by SNR, but fewer provide a controlled comparison of clean accuracy,
low-SNR degradation, narrowband jamming, broadband jamming, carrier-frequency
offset, multipath, impulsive noise, ablation against a compressed-feature
kernel, and inference latency in one reproducible workflow. Recent Results in
Engineering work has shown that RadioML-based AMR papers are expected to
include explicit ablations, efficiency metrics, and low-SNR robustness claims
[31]. RSC-Bench extends this style by making the robustness matrix itself the
primary contribution and by including a same-feature quantum/classical kernel
ablation to avoid unsupported claims.

### 2.3 Robust RFML, jamming, and adversarial vulnerability

Robustness is central to RF machine learning because the wireless channel and
the attacker both act directly on the received waveform. Prior studies have
shown that deep RF classifiers can be sensitive to adversarial perturbations,
over-the-air attacks, channel-aware attacks, and wireless jamming [15-20].
Noise-mismatch work similarly shows that models trained under one impairment
distribution may degrade sharply under another [14]. These studies support the
need for a stress-test protocol that reports failure modes rather than only
clean benchmark scores.

The present paper is not an adversarial-attack paper in the strict white-box or
black-box optimization sense. Instead, it provides a reproducible engineering
stress benchmark using physically interpretable perturbations: low SNR,
narrowband jamming, broadband jamming, carrier-frequency offset, multipath,
and impulsive noise.

### 2.4 Quantum-inspired kernels for classical RF data

Quantum feature maps can be viewed as nonlinear embeddings into Hilbert spaces,
and quantum kernels can be used in a support-vector-machine framework [21,
22]. This perspective makes quantum-inspired kernels attractive as compact
nonlinear classifiers for structured signal features. At the same time,
quantum-machine-learning theory warns that advantage claims must be supported
by careful comparisons against classical learners and classical kernels [23,
24]. In particular, if a quantum feature map is applied to low-dimensional
classical features, a classical RBF kernel on the same features may perform
similarly.

The research gap addressed here is therefore not whether a quantum kernel can
be constructed, but whether a simulated quantum feature-map kernel improves RF
classification or robustness when compared directly to a PCA-RBF SVM on the
same compressed features. This ablation is included to prevent unsupported
quantum-advantage claims.

### 2.5 Position of this study

Compared with prior work, this study is positioned as a reproducible
robustness benchmark rather than a single-model performance paper. The novelty
is the named RSC-Bench protocol and the combination of:

1. a stress-condition protocol covering six degraded RF conditions beyond
   clean evaluation;
2. side-by-side comparison of classical ML, raw-IQ CNN, simulated
   quantum-feature-map kernel SVM, and a same-feature PCA-RBF ablation;
3. full RadioML2016.10A GPU validation in addition to a controlled synthetic
   dataset;
4. explicit reporting of failure modes and robustness drops; and
5. complexity and inference-latency reporting to support engineering model
   selection.

## 3. Contributions

This study contributes:

1. A reproducible RF/IQ signal-classification workflow covering synthetic IQ
   data and the public RadioML2016.10A benchmark.
2. A robustness protocol covering clean, low-SNR, narrowband-jam,
   broadband-jam, frequency-offset, multipath, and impulsive-noise conditions.
3. A transparent comparison of engineered-feature classical ML, raw-IQ CNN,
   simulated quantum feature-map kernel SVM, and same-feature PCA-RBF SVM
   ablation baselines.
4. A direct test of the quantum-inspired feature map against a classical RBF
   kernel on the same five-dimensional PCA representation.
5. A computational complexity and inference-latency analysis comparing
   parameter counts and per-sample execution time.
6. Full GPU benchmark results on RadioML2016.10A with clean performance,
   stress-condition metrics, robustness-drop tables, and publication-quality
   figures.
7. A cautious interpretation of the quantum-inspired component that avoids
   unsupported quantum-advantage claims.

## 4. Materials And Methods

### 4.1 Datasets and signal representation

A discrete-time received complex-valued IQ signal sample is represented as:

`x[n] = I[n] + jQ[n], n in {0, 1, ..., N-1}`

where `I[n]` and `Q[n]` represent the in-phase and quadrature channels,
respectively, and `N = 128` is the sequence length. To reduce amplitude-scale
differences, the signal vector is normalized to unit average power before
feature extraction or CNN training.

The controlled synthetic dataset uses eight modulation families: BPSK, QPSK,
8PSK, QAM16, QAM64, BFSK, AM-DSB, and FM. The scaled synthetic run contains
2,000 samples per class, giving 16,000 total examples. The synthetic generator
applies randomized phase rotation, carrier offset, optional multipath, and
additive white Gaussian noise across SNR values of -6, 0, 6, 12, and 18 dB.

The public benchmark validation uses RadioML2016.10A. The converted clean NPZ
contains 220,000 examples, 11 modulation classes, and SNR values from -20 to
18 dB in 2 dB steps. The classes are 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK,
PAM4, QAM16, QAM64, QPSK, and WBFM. The raw RadioML file is not redistributed;
users must obtain it from the original dataset provider or validated public
mirror and comply with applicable license terms.

### 4.2 Stress conditions and degradation models

To evaluate classifier robustness in contested spectrum environments, a set of
repeatable signal stress operators is applied to each held-out test signal:

- Low SNR: additive complex Gaussian noise is applied to evaluate operation at
  reduced signal-to-noise ratio.
- Narrowband jamming: a high-amplitude sinusoidal tone with randomized phase
  and normalized frequency is added to the received IQ sequence.
- Broadband jamming: high-power broadband noise is added to approximate
  wideband interference.
- Carrier-frequency offset: a progressive complex phase rotation is applied to
  the IQ sequence.
- Multipath fading: delayed and phase-rotated copies of the signal are added
  with fixed sample delays and random path phases.
- Impulsive noise: sparse high-amplitude impulses are injected into the signal.

These stressors are intentionally interpretable. They do not exhaust all
possible channels or attacks, but they provide repeatable degradation
conditions that can be audited and reproduced.

### 4.3 Classical and deep-learning models

Classical baselines use a 15-dimensional engineered RF feature vector derived
from IQ statistics, amplitude statistics, phase statistics, spectral centroid,
spectral bandwidth, and FFT peak-to-mean ratio. The evaluated classical models
are Logistic Regression, Random Forest, and RBF-SVM. Logistic Regression uses
standardized features with maximum 1,500 iterations and regularization
parameter `C = 2.0`; Random Forest uses 250 trees, random seed 2026, and
minimum leaf size 2; RBF-SVM uses standardized features with `C = 8.0` and
`gamma = scale`.

The deep-learning baseline is a compact one-dimensional convolutional neural
network trained directly on raw IQ samples. It uses three Conv1D blocks with
channel widths 32, 64, and 128; kernel sizes 7, 5, and 3; batch
normalization; ReLU activations; pooling; adaptive average pooling; dropout
0.15; and a linear classification head. In the full RadioML run, the CNN was
trained on CUDA for 40 epochs with batch size 512, AdamW optimization,
learning rate 0.001, weight decay 0.0001, random seed 2026, and best-epoch
restoration. The final restored model came from epoch 18.

### 4.4 Quantum-inspired kernel and classical ablation

The quantum-inspired baseline is a simulated quantum feature-map kernel SVM. A
15-dimensional engineered statistical feature vector is standardized and
projected to five dimensions using principal component analysis (PCA). The
resulting five-dimensional vector is scaled to angular values in `[-pi, pi]`
and mapped to a complex `2^5`-dimensional simulated statevector using
single-qubit angle features and pairwise phase interactions with entanglement
strength 0.35. The precomputed-kernel SVM uses `C = 4.0` and is trained on a
fidelity-style precomputed kernel:

`K(x_i, x_j) = |<psi(theta_i)|psi(theta_j)>|^2`

This is a simulated quantum-inspired feature map, not a hardware
implementation and not a quantum-advantage claim.

To isolate whether the quantum feature map contributes beyond a standard
kernel, a classical PCA-RBF SVM ablation baseline is trained on the identical
five-dimensional PCA features. If the quantum feature map adds meaningful
representational value, it should outperform this same-feature classical
kernel. If it does not, the quantum-inspired embedding should be reported as a
negative or neutral baseline rather than as a superior method.

### 4.5 RSC-Bench protocol and formal pseudocode

RSC-Bench is the named evaluation protocol used in this study. Its purpose is
to prevent three common weaknesses in applied RFML papers: reporting clean
accuracy without degradation evidence, comparing a quantum-inspired model only
against weak baselines, and omitting computational cost.

Algorithm 1. RSC-Bench robustness evaluation protocol.

Input: clean dataset `D = {(x_i, y_i)}`, model set `M`, stress-operator set
`S`, stratified split rule `pi`, metric set `G = {accuracy, macro-F1}`,
feature extractor `phi`, and random seed `r`.

Output: clean metrics, stress metrics, robustness-drop table, same-feature
kernel ablation table, and complexity table.

1. Normalize each IQ sequence `x_i` to unit average power.
2. Split `D` into stratified training and held-out test sets using `pi` and
   seed `r`.
3. For each model `m` in `M`, train `m` on the clean training set or its
   feature representation `phi(D_train)`.
4. Evaluate each trained model on the clean held-out test set and record all
   metrics in `G`.
5. For each stress operator `s` in `S`, transform the held-out test signals as
   `D_test_s = s(D_test)`.
6. Evaluate every trained model on `D_test_s` without retraining.
7. Compute robustness drop for each metric as
   `drop_s = 100 * (metric_clean - metric_s) / metric_clean`.
8. Train the simulated quantum feature-map kernel SVM on five PCA components
   of `phi(D_train)`.
9. Train a classical PCA-RBF SVM on the same five PCA components.
10. Compare the two kernel models on clean and stressed test subsets to test
    whether the quantum feature map adds value beyond the same compressed
    classical feature space.
11. Record training time, parameter or support-vector count, and per-sample
    inference latency for each model family.
12. Export reproducible CSV tables, figures, and repository commands.

### 4.6 Computational complexity analysis

Let `N_tr` and `N_te` denote the numbers of training and held-out test
examples, `L = 128` denote IQ sequence length, `F = 15` denote the engineered
feature dimension, `S` denote the number of stress operators, `M` denote the
number of model families, `q = 5` denote the number of simulated qubits, and
`n_q` denote the compact training subset used by the quantum-kernel model.
Stress generation has time complexity `O(S N_te L)` and stores one transformed
test array per stressor when materialized. Engineered-feature extraction has
time complexity `O((N_tr + N_te) L log L)` because FFT-derived statistics are
included, with feature memory `O((N_tr + N_te) F)`.

For the classical models, Logistic Regression training scales approximately as
`O(I N_tr F C)` for `I` optimizer iterations and `C` modulation classes, Random
Forest training scales with the number of trees and split evaluations, and
RBF-SVM training can scale between quadratic and cubic behavior in `N_tr`
depending on the solver and support-vector count. The raw-IQ CNN has
per-epoch training complexity proportional to the number of convolutional
multiply-add operations across `N_tr` sequences and has inference complexity
linear in the number of learned convolutional filters and sequence length.

For the simulated quantum feature-map kernel, state construction costs
`O(n_q 2^q)`, training-kernel construction costs `O(n_q^2 2^q)`, and
precomputed SVM training can scale up to `O(n_q^3)` in the worst case. The
kernel matrix requires `O(n_q^2)` memory and the state table requires
`O(n_q 2^q)` memory. These costs motivate the compact 120 train-per-class and
80 test-per-class RadioML configuration used for the quantum-kernel baseline.

### 4.7 Evaluation protocol

All models use stratified held-out evaluation. Clean performance is measured
using accuracy and macro F1. Robustness is measured by evaluating the trained
model on held-out examples transformed by each stress condition and reporting
the percentage drop from clean accuracy and macro F1. The full RadioML run used
30,000 training and 10,000 test examples for the classical baselines, 80,000
training and 20,000 test examples for the raw-IQ CNN, and a compact 120
train-per-class / 80 test-per-class configuration for the simulated quantum
feature-map kernel SVM.

## 5. Results

### 5.1 Synthetic dataset results

The controlled synthetic dataset contains 2,000 samples per class and 16,000
total examples across eight modulation families and five SNR levels. The clean
classification table is provided in:

`manuscript_assets/tables/table_synthetic_clean_performance.csv`

With 2,000 samples per class, the raw-IQ CNN achieved the highest clean
held-out accuracy (0.654), followed by Random Forest (0.592), RBF-SVM (0.570),
Logistic Regression (0.554), the simulated quantum feature-map kernel SVM
(0.410), and the classical PCA-RBF SVM ablation baseline (0.408). The CNN
maintained the lowest mean stress-induced accuracy drop across the synthetic
robustness protocol, but remained vulnerable to narrowband jamming and
frequency offset. The PCA-RBF SVM and simulated quantum-kernel SVM achieved
statistically indistinguishable clean accuracy (0.408 vs. 0.410), showing that
the quantum feature map did not provide measurable benefit over a classical
kernel on the same compressed features.

### 5.2 RadioML2016.10A clean performance

The full GPU RadioML benchmark is the primary public-dataset evidence for this
study. The clean classification table is provided in:

`manuscript_assets/tables/table_radioml2016_full_clean_performance.csv`

On RadioML2016.10A, the raw-IQ CNN achieved the strongest clean performance
with 0.5114 accuracy and 0.5231 macro-F1 on 20,000 held-out examples. Random
Forest achieved 0.4846 accuracy and 0.4963 macro-F1; RBF-SVM achieved 0.4739
accuracy and 0.4887 macro-F1; Logistic Regression achieved 0.4344 accuracy and
0.4345 macro-F1. The simulated quantum feature-map kernel SVM achieved 0.3091
accuracy and 0.2953 macro-F1 on its compact evaluation subset.

### 5.3 RadioML2016.10A robustness under degraded conditions

Robustness metrics and drop rates under degraded channel stressors are
provided in:

`manuscript_assets/tables/table_radioml2016_full_robustness_metrics.csv`

`manuscript_assets/tables/table_radioml2016_full_robustness_drop.csv`

All models degraded under stress conditions. For the raw-IQ CNN, the largest
accuracy drops were observed under narrowband jamming (82.08 percent drop),
frequency offset (81.61 percent), and impulsive noise (80.28 percent). The CNN
was less affected by broadband jamming (33.64 percent drop) and low SNR (52.02
percent drop) relative to its most severe failure modes. Random Forest also
degraded sharply under narrowband jamming, while RBF-SVM collapsed toward
near-chance performance across all stressors. These results show that model
selection based only on clean-set accuracy would miss important deployment
risks.

### 5.4 Quantum-inspired kernel versus PCA-RBF ablation

The simulated quantum feature-map kernel baseline did not outperform the
classical or CNN baselines. Its clean RadioML accuracy was 0.3091, and stress
conditions reduced performance toward near-chance levels. On the 2,000-sample
synthetic dataset, the classical PCA-RBF SVM achieved 0.408 clean accuracy
compared with 0.410 for the quantum kernel SVM, a difference within practical
noise. This ablation indicates that the simulated quantum circuit does not
confer a classification or robustness advantage over a standard kernel method
operating on the same five-dimensional PCA feature space.

### 5.5 Computational complexity and inference latency

Computational-complexity metrics are provided in:

`manuscript_assets/tables/complexity_metrics.csv`

The raw-IQ CNN required the longest training time among the lightweight models
in the synthetic run, but achieved fast inference at 0.155 ms per sample.
Logistic Regression was the most lightweight model. The simulated
quantum-kernel SVM incurred higher inference latency than the PCA-RBF SVM due
to statevector computation overhead, while offering no compensating accuracy
benefit. This cost-benefit analysis further supports the conclusion that the
quantum-inspired kernel should be treated as a transparent comparator rather
than as a deployable advantage.

## 6. Discussion

The full public benchmark supports the central engineering premise: clean
classification accuracy is insufficient for RF model evaluation. A model may
rank well under clean conditions while failing under a particular stressor. The
raw-IQ CNN produced the best clean RadioML result, yet it degraded sharply
under narrowband jamming and carrier-frequency offset. Classical models also
showed stress-specific collapse, especially under jamming and low-SNR variants.

The sensitivity of the raw-IQ CNN to carrier-frequency offset is explainable:
CFO introduces progressive phase rotation over time, which can invalidate the
local temporal patterns learned by standard one-dimensional convolutional
filters. Narrowband jamming introduces a high-energy localized tone that can
dominate the received waveform and contaminate learned filters unless explicit
spectral filtering, augmentation, or channel-compensation strategies are used.

The gap between clean and degraded performance is important for contested
spectrum applications. Robust RF classifiers should be reported with both
nominal and stress-condition metrics, including the specific perturbations that
cause failure. The robustness-drop tables provide a compact way to compare
models under this requirement.

The quantum-inspired component should be interpreted carefully. The results do
not justify claims of quantum superiority. Instead, they show how a simulated
quantum feature-map kernel can be included transparently in an engineering
benchmark and compared against practical alternatives. The classical PCA-RBF
SVM ablation provides the strongest evidence for this conclusion: when both
classifiers receive identical five-dimensional PCA features, the quantum
feature map's exponential Hilbert-space embedding does not improve accuracy or
robustness over a standard RBF kernel.

## 7. Limitations

This study is simulation and public-benchmark based. It does not use live RF
captures, hardware-in-the-loop evaluation, over-the-air adversarial testing, or
classified defense data. RadioML2016.10A is a useful public benchmark, but it
does not represent every operational RF channel or emitter environment. The
stress transformations are controlled and reproducible rather than exhaustive.
The simulated quantum feature-map kernel is evaluated on a compact subset
because kernel methods scale poorly with the number of training examples.

The paper therefore should be interpreted as a reproducible engineering
benchmark and robustness audit, not as a claim of operational deployment
readiness or quantum advantage.

## 8. Conclusion

This work establishes RSC-Bench, a reproducible evaluation protocol for
robustness-aware
RF signal classification under degraded spectrum conditions. The full
RadioML2016.10A GPU benchmark shows that the raw-IQ CNN has the strongest clean
performance among the tested models, but also reveals large stress-condition
failure modes. Classical models and the simulated quantum-inspired kernel
provide useful comparators, while the quantum-inspired result is best framed as
a transparent baseline rather than a quantum-advantage result. The main
engineering contribution is a reproducible robustness protocol and evidence
package for AI-based RF classification in contested-spectrum environments.

## 9. Submission Integrity And AI-Assisted Writing Disclosure

All numerical results reported in the manuscript are derived from the CSV
tables and figures generated by the repository scripts. No unresolved DOIs,
unresolved funding statements, unresolved ethics statements, or unsupported
quantum-advantage claims are intentionally retained. The raw RadioML2016.10A
dataset is not redistributed; the manuscript reports derived metrics and
provides commands for independent reproduction by users with lawful dataset
access. AI-assisted drafting and code-generation tools were used for editorial
drafting, code scaffolding, and consistency checking; the author reviewed,
executed, and verified the analyses, metrics, and final text.

## References

1. O. A. Dobre, A. Abdi, Y. Bar-Ness, and W. Su, "Survey of automatic modulation classification techniques: Classical approaches and new trends," IET Communications, vol. 1, no. 2, pp. 137-156, 2007, doi: 10.1049/iet-com:20050176.
2. D. Grimaldi, S. Rapuano, and L. De Vito, "An automatic digital modulation classifier for measurement on telecommunication networks," IEEE Transactions on Instrumentation and Measurement, vol. 56, no. 5, pp. 1711-1720, 2007, doi: 10.1109/TIM.2007.895675.
3. T. J. O'Shea, J. Corgan, and T. C. Clancy, "Convolutional radio modulation recognition networks," in Engineering Applications of Neural Networks, pp. 213-226, 2016, doi: 10.1007/978-3-319-44188-7_16.
4. N. E. West and T. J. O'Shea, "Deep architectures for modulation recognition," in 2017 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp. 1-6, 2017, doi: 10.1109/DySPAN.2017.7920754.
5. T. J. O'Shea, T. Roy, and T. C. Clancy, "Over-the-air deep learning based radio signal classification," IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, 2018, doi: 10.1109/JSTSP.2018.2797022.
6. F. Meng, P. Chen, L. Wu, and X. Wang, "Automatic modulation classification: A deep learning enabled approach," IEEE Transactions on Vehicular Technology, vol. 67, no. 11, pp. 10760-10772, 2018, doi: 10.1109/TVT.2018.2868698.
7. S. Rajendran, W. Meert, D. Giustiniano, V. Lenders, and S. Pollin, "Deep learning models for wireless signal classification with distributed low-cost spectrum sensors," IEEE Transactions on Cognitive Communications and Networking, vol. 4, no. 3, pp. 433-445, 2018, doi: 10.1109/TCCN.2018.2835460.
8. Y. Wang, M. Liu, J. Yang, and G. Gui, "Data-driven deep learning for automatic modulation recognition in cognitive radios," IEEE Transactions on Vehicular Technology, vol. 68, no. 4, pp. 4074-4077, 2019, doi: 10.1109/TVT.2019.2900460.
9. P. Qi, S. Zheng, S. Chen, and X. Yang, "Fusion methods for CNN-based automatic modulation classification," IEEE Access, vol. 7, pp. 66496-66504, 2019, doi: 10.1109/ACCESS.2019.2918136.
10. T. Huynh-The, Q. V. Pham, T. V. Nguyen, T. T. Nguyen, R. Ruby, M. Zeng, and D. S. Kim, "Automatic modulation classification: A deep architecture survey," IEEE Access, vol. 9, pp. 142950-142971, 2021, doi: 10.1109/ACCESS.2021.3120419.
11. F. Shi, Z. Hu, C. Yue, and Z. Shen, "Combining neural networks for modulation recognition," Digital Signal Processing, vol. 120, 103264, 2022, doi: 10.1016/j.dsp.2021.103264.
12. C. A. Harper, M. A. Thornton, and E. C. Larson, "Automatic modulation classification with deep neural networks," Electronics, vol. 12, no. 18, 3962, 2023, doi: 10.3390/electronics12183962.
13. X. Tian et al., "A survey on deep learning enabled automatic modulation classification methods: Data representations, model structures, and regularization techniques," Signal Processing, vol. 242, 110444, 2026, doi: 10.1016/j.sigpro.2025.110444.
14. Z. Wu, S. Zhou, Z. Yin, B. Ma, and Z. Yang, "Robust automatic modulation classification under varying noise conditions," IEEE Access, vol. 5, pp. 19733-19741, 2017, doi: 10.1109/ACCESS.2017.2746140.
15. M. Sadeghi and E. G. Larsson, "Adversarial attacks on deep-learning based radio signal classification," IEEE Wireless Communications Letters, vol. 8, no. 1, pp. 213-216, 2019, doi: 10.1109/LWC.2018.2867459.
16. B. Flowers, R. M. Buehrer, and W. C. Headley, "Evaluating adversarial evasion attacks in the context of wireless communications," IEEE Transactions on Information Forensics and Security, vol. 15, pp. 1102-1113, 2020, doi: 10.1109/TIFS.2019.2934069.
17. B. Kim, Y. E. Sagduyu, K. Davaslioglu, T. Erpek, and S. Ulukus, "Over-the-air adversarial attacks on deep learning based modulation classifier over wireless channels," in 2020 54th Annual Conference on Information Sciences and Systems (CISS), pp. 1-6, 2020, doi: 10.1109/CISS48834.2020.1570617416.
18. B. Kim, Y. E. Sagduyu, K. Davaslioglu, T. Erpek, and S. Ulukus, "Channel-aware adversarial attacks against deep learning-based wireless signal classifiers," IEEE Transactions on Wireless Communications, vol. 21, no. 6, pp. 3868-3880, 2022, doi: 10.1109/TWC.2021.3124855.
19. T. Erpek, Y. E. Sagduyu, and Y. Shi, "Deep learning for launching and mitigating wireless jamming attacks," IEEE Transactions on Cognitive Communications and Networking, vol. 5, no. 1, pp. 2-14, 2019, doi: 10.1109/TCCN.2018.2884910.
20. Y. Shi, Y. E. Sagduyu, T. Erpek, K. Davaslioglu, Z. Lu, and J. H. Li, "Adversarial deep learning for cognitive radio security: Jamming attack and defense strategies," in IEEE ICC Workshops, pp. 1-6, 2018, doi: 10.1109/ICCW.2018.8403655.
21. V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," Nature, vol. 567, pp. 209-212, 2019, doi: 10.1038/s41586-019-0980-2.
22. M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," Physical Review Letters, vol. 122, no. 4, 040504, 2019, doi: 10.1103/PhysRevLett.122.040504.
23. H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," Nature Communications, vol. 12, 2631, 2021, doi: 10.1038/s41467-021-22539-9.
24. S. Jerbi, L. J. Fiderer, H. P. Nautrup, J. M. Kubler, H. J. Briegel, and V. Dunjko, "Quantum machine learning beyond kernel methods," Nature Communications, vol. 14, 517, 2023, doi: 10.1038/s41467-023-36159-y.
25. J. Biamonte et al., "Quantum machine learning," Nature, vol. 549, pp. 195-202, 2017, doi: 10.1038/nature23474.
26. T. Huynh-The, C. H. Hua, Q. V. Pham, and D. S. Kim, "MCNet: An efficient CNN architecture for robust automatic modulation classification," IEEE Communications Letters, vol. 24, no. 4, pp. 811-815, 2020, doi: 10.1109/LCOMM.2020.2968030.
27. A. P. Hermawan, R. R. Ginanjar, D. S. Kim, and J. M. Lee, "CNN-based automatic modulation classification for beyond 5G communications," IEEE Communications Letters, vol. 24, no. 5, pp. 1038-1041, 2020, doi: 10.1109/LCOMM.2020.2970922.
28. N. Wang, Y. Liu, L. Ma, Y. Yang, and H. Wang, "Multidimensional CNN-LSTM network for automatic modulation classification," Electronics, vol. 10, no. 14, 1649, 2021, doi: 10.3390/electronics10141649.
29. L. Weng, Y. He, J. Peng, J. Zheng, and X. Li, "Deep cascading network architecture for robust automatic modulation classification," Neurocomputing, vol. 455, pp. 308-324, 2021, doi: 10.1016/j.neucom.2021.05.010.
30. Z. Lyu, Y. Wang, W. Li, L. Guo, J. Yang, J. Sun, M. Liu, and G. Gui, "Robust automatic modulation classification based on convolutional and recurrent fusion network," Physical Communication, vol. 43, 101213, 2020, doi: 10.1016/j.phycom.2020.101213.
31. N. El-Haryqy, A. Kharbouche, H. Ouamna, Z. Madini, and Y. Zouine, "Improved automatic modulation recognition using deep learning with additive attention," Results in Engineering, vol. 26, 104783, 2025, doi: 10.1016/j.rineng.2025.104783.
32. A. Kumar, M. S. Chaudhari, and S. Majhi, "Automatic modulation classification for OFDM systems using bi-stream and attention-based CNN-LSTM model," IEEE Communications Letters, 2024, doi: 10.1109/LCOMM.2023.3348512.
33. C. Zhao, J. Chen, X. Huang, and Z. Wu, "A cross-scale embedding based fusion transformer for automatic modulation recognition," IEEE Communications Letters, vol. 28, no. 1, pp. 68-72, 2024, doi: 10.1109/LCOMM.2023.3331265.
34. H. Xing, X. Zhang, S. Chang, J. Ren, Z. Zhang, J. Xu, and S. Cui, "Joint signal detection and automatic modulation classification via deep learning," IEEE Transactions on Wireless Communications, vol. 23, no. 11, pp. 17129-17142, 2024, doi: 10.1109/TWC.2024.3450972.
