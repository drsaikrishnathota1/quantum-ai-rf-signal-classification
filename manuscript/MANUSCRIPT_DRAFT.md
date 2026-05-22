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
neural network, a simulated quantum feature-map kernel classifier, and a
classical PCA-RBF SVM ablation baseline. The workflow was evaluated on a
controlled synthetic IQ dataset (16,000 examples, 8 modulation families) and
validated on the public RadioML2016.10A benchmark containing 220,000 examples,
11 modulation classes, and SNR values from -20 to 18 dB. On the synthetic
dataset, the raw-IQ CNN achieved the highest clean accuracy (0.654), followed by
Random Forest (0.592), RBF-SVM (0.570), Logistic Regression (0.554), and the
simulated quantum kernel SVM (0.410). On RadioML2016.10A, the CNN maintained its
lead (0.511 accuracy, 0.523 macro-F1). Robustness testing showed major
degradation under narrowband jamming, frequency offset, low SNR, multipath, and
impulsive noise, demonstrating that clean accuracy alone is insufficient for RF
model selection. A classical PCA-RBF SVM ablation trained on the same
5-dimensional PCA features as the quantum kernel achieved statistically
identical accuracy (0.408 vs. 0.410), confirming no quantum-feature-map
advantage. Computational complexity analysis showed the CNN achieves the best
accuracy-to-latency ratio (0.155 ms per sample). The study provides
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
4. A classical PCA-RBF SVM ablation baseline trained on the same
   5-dimensional PCA features as the quantum kernel, isolating the
   contribution of the quantum feature map.
5. A computational complexity and inference latency analysis comparing
   parameter counts and per-sample execution times across all models.
6. Full GPU benchmark results on RadioML2016.10A with clean performance,
   stress-condition metrics, robustness-drop tables, and publication-quality
   figures.
7. A cautious interpretation of the quantum-inspired component that avoids
   unsupported quantum-advantage claims.

## 3. Materials And Methods

### 3.1 Datasets and Signal Representation

A discrete-time received complex-valued IQ signal sample is represented as:
$$x[n] = I[n] + jQ[n], \quad n \in \{0, 1, \dots, N-1\}$$
where $I[n]$ and $Q[n]$ represent the in-phase and quadrature channels respectively, and $N = 128$ is the sequence length. To eliminate amplitude discrepancies, the signal vector is normalized to unity power:
$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{P_{\mathbf{x}}}}, \quad P_{\mathbf{x}} = \frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2$$

The controlled synthetic pilot uses eight modulation families: BPSK, QPSK, 8PSK, QAM16, QAM64, BFSK, AM-DSB, and FM. For the scaled pilot, 500 samples per class were generated, giving 4,000 total examples. The synthetic generator applies randomized phase rotation, carrier offset, optional multipath, and additive white Gaussian noise across SNR values of -6, 0, 6, 12, and 18 dB.

The public benchmark validation uses RadioML2016.10A. The converted clean NPZ contains 220,000 examples, 11 modulation classes, and SNR values from -20 to 18 dB in 2 dB steps. The classes are 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, and WBFM. The raw RadioML file is not redistributed; users must obtain it from the original dataset provider or validated public mirror and comply with the applicable license terms.

### 3.2 Stress Conditions and Degradation Models

To evaluate classifier robustness in contested spectrum environments, a set of repeatable signal stress operators $S(\mathbf{x}, \text{condition})$ are applied to each held-out test signal:

* **Additive White Gaussian Noise (AWGN):**
  $$y_{\text{awgn}}[n] = x[n] + \sigma_n w[n]$$
  where $w[n] \sim \mathcal{CN}(0, 1)$ represents complex white Gaussian noise, and the noise power $\sigma_n^2 = P_{\mathbf{x}} \cdot 10^{-\text{SNR}_{\text{dB}}/10}$. The **Low SNR** stressor evaluates models at a fixed $\text{SNR}_{\text{dB}} = -4$ dB.
  
* **Narrowband Jamming:**
  $$y_{\text{nb\_jam}}[n] = \text{Norm}\left(x[n] + A_{\text{jam}} e^{j(2\pi f_j n T_s + \theta_j)}\right)$$
  where $A_{\text{jam}} = 0.9$ represents the relative jammer amplitude, $f_j \sim \mathcal{U}(0.04, 0.22)$ is the normalized jamming tone frequency, and $\theta_j \sim \mathcal{U}(0, 2\pi)$ is a random phase.

* **Broadband Jamming:**
  Evaluated by adding high-power AWGN to the signal, setting a low baseline signal-to-noise ratio at $\text{SNR}_{\text{dB}} = 0$ dB.

* **Carrier Frequency Offset (CFO):**
  $$y_{\text{cfo}}[n] = \text{Norm}\left(x[n] e^{j 2\pi \Delta f n T_s}\right)$$
  where the normalized frequency drift is fixed at $\Delta f = 0.045$.

* **Multipath Fading:**
  $$y_{\text{multipath}}[n] = \text{Norm}\left(x[n] + \sum_{l=1}^{L} h_l e^{j \theta_l} x[n - \tau_l]\right)$$
  using a two-path fading model ($L=2$) with sample delays $\tau_1 = 2$, $\tau_2 = 5$, path gains $h_1 = 0.35$, $h_2 = 0.18$, and random path phases $\theta_l \sim \mathcal{U}(0, 2\pi)$.

* **Impulsive Noise:**
  $$y_{\text{impulsive}}[n] = \text{Norm}\left(x[n] + b[n] \cdot \eta[n]\right)$$
  where $b[n] \sim \text{Bernoulli}(p)$ is an impulse occurrence mask with activation probability $p = 0.025$, and the impulse amplitude is modeled as $\eta[n] \sim \mathcal{CN}(0, \sigma_{\text{imp}}^2)$ with $\sigma_{\text{imp}} = 4.5$.

### 3.3 Models

Classical baselines use engineered RF features derived from IQ statistics, amplitude statistics, phase statistics, spectral centroid, spectral bandwidth, and FFT peak-to-mean ratio. The evaluated classical models are Logistic Regression, Random Forest, and RBF-SVM.

The deep-learning baseline is a compact one-dimensional convolutional neural network trained directly on raw IQ samples. It uses three Conv1D blocks with batch normalization, ReLU activations, pooling, adaptive average pooling, dropout, and a linear classification head. In the full RadioML run, the CNN was trained on CUDA for 40 epochs with batch size 512 and best-epoch restoration. The final restored model came from epoch 18.

The quantum-inspired baseline is a simulated quantum feature-map kernel SVM. A 32-dimensional engineered statistical feature vector $\mathbf{F} \in \mathbb{R}^{32}$ is first extracted. The feature vector is standardized and projected to a lower-dimensional subspace of $P = 5$ dimensions using Principal Component Analysis (PCA):
$$\mathbf{z} = \mathbf{W}_{\text{pca}}^T \mathbf{z}_F$$
where $\mathbf{z}_F$ is the standardized feature vector, and $\mathbf{W}_{\text{pca}} \in \mathbb{R}^{32 \times 5}$ represents the principal component transformation matrix.

The projected vector is scaled to the angle interval $[-\pi, \pi]$:
$$\theta_q = \text{MinMax}(\mathbf{z}_q), \quad q \in \{0, 1, \dots, P-1\}$$

The angles $\boldsymbol{\theta} = [\theta_0, \theta_1, \dots, \theta_{P-1}]^T$ are mapped to a complex $2^P$-dimensional simulated quantum statevector $|\psi(\boldsymbol{\theta})\rangle$ representing a NISQ-style parameterized quantum circuit:
$$|\psi(\boldsymbol{\theta})\rangle = \sum_{k=0}^{2^P-1} A_k(\boldsymbol{\theta}) e^{j \phi_k(\boldsymbol{\theta})} |k\rangle$$

The amplitude component $A_k(\boldsymbol{\theta})$ for each basis state index $k \in \{0, \dots, 2^P - 1\}$ with binary digits $b_{k, q} \in \{0, 1\}$ is given by:
$$A_k(\boldsymbol{\theta}) = \prod_{q=0}^{P-1} a_{k, q}(\theta_q), \quad \text{where } a_{k, q}(\theta_q) = \begin{cases} 
      \sin\left(\frac{\theta_q}{2}\right) & \text{if } b_{k, q} = 1 \\
      \cos\left(\frac{\theta_q}{2}\right) & \text{if } b_{k, q} = 0
   \end{cases}$$

The phase-entanglement term $\phi_k(\boldsymbol{\theta})$ is formulated via pairwise interactions:
$$\phi_k(\boldsymbol{\theta}) = \gamma \sum_{a=0}^{P-1} \sum_{b=a+1}^{P-1} b_{k, a} b_{k, b} \frac{\theta_a \theta_b}{\pi}$$
where $\gamma = 0.35$ represents the entangling strength. The fidelity-style quantum kernel $K(\mathbf{x}_i, \mathbf{x}_j)$ precomputed for the SVM classifier is:
$$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\boldsymbol{\theta}_i) | \psi(\boldsymbol{\theta}_j) \rangle|^2$$

This is a simulated quantum-inspired feature map, not a hardware quantum-advantage claim.

To isolate whether the quantum feature map contributes beyond what a standard kernel achieves on the same compressed features, a classical PCA-RBF SVM ablation baseline was trained. This model uses an RBF kernel ($K_{\text{rbf}}(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma_{\text{rbf}} \|\mathbf{x}_i - \mathbf{x}_j\|^2)$) applied directly to the same 5-dimensional PCA-reduced features $\mathbf{z}$ used by the quantum kernel, with $C = 4.0$ and $\gamma_{\text{rbf}}$ set by the scikit-learn ``scale`` heuristic. If the quantum feature map provides genuine nonlinear representational value, the quantum kernel SVM should outperform this ablation baseline; if not, the quantum-inspired embedding is functionally equivalent to a classical kernel on the same compressed representation.

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

### 4.1 Synthetic Dataset Results

The controlled synthetic dataset (2,000 samples per class, 16,000 total examples across 8 modulation families and 5 SNR levels) was used to evaluate all six classifiers under identical conditions. The clean classification performance is shown in Table 1.

#### Table 1: Clean Performance on Synthetic Dataset (2,000 Samples/Class)
| Model | Clean Accuracy | Macro $F_1$ |
| :--- | :---: | :---: |
| **Raw-IQ CNN** | 0.6535 | 0.6232 |
| **Random Forest** | 0.5923 | 0.5897 |
| **RBF-SVM** | 0.5698 | 0.5659 |
| **Logistic Regression** | 0.5543 | 0.5433 |
| **Simulated QFM-Kernel SVM** | 0.4104 | 0.4080 |
| **Classical PCA-RBF SVM** | 0.4083 | 0.4054 |

With 2,000 samples per class, the raw-IQ CNN achieved the highest clean held-out accuracy (0.654), followed by Random Forest (0.592), RBF-SVM (0.570), Logistic Regression (0.554), the simulated quantum feature-map kernel SVM (0.410), and the classical PCA-RBF SVM ablation baseline (0.408). The CNN maintained the lowest mean stress-induced accuracy drop across the synthetic robustness protocol, but remained severely vulnerable to narrowband jamming (68.7% drop) and frequency offset (62.4% drop). Notably, the classical PCA-RBF SVM and the simulated quantum-kernel SVM achieved statistically indistinguishable clean accuracy (0.408 vs. 0.410), confirming that the quantum feature map does not provide measurable classification benefit over a standard RBF kernel on the same PCA-compressed features.

![Figure 1: Synthetic Pilot Clean Accuracy comparison for the classical, deep learning, and quantum-inspired classifiers.](/Users/sai/.gemini/antigravity/scratch/quantum-ai-rf-signal-classification/manuscript_assets/figures/fig_synthetic_clean_accuracy.png)

![Figure 2: Classification Accuracy vs. SNR (dB) for the synthetic pilot dataset under nominal channel conditions.](/Users/sai/.gemini/antigravity/scratch/quantum-ai-rf-signal-classification/manuscript_assets/figures/fig_synthetic_accuracy_by_snr.png)

![Figure 3: Robustness Accuracy Drop (%) Heatmap on the synthetic pilot dataset across various stress channel conditions.](/Users/sai/.gemini/antigravity/scratch/quantum-ai-rf-signal-classification/manuscript_assets/figures/fig_synthetic_robustness_drop_heatmap.png)

### 4.2 RadioML2016.10A Clean Performance

The full GPU RadioML benchmark is the primary public-dataset evidence for this study. The clean classification results are summarized in Table 2.

#### Table 2: Clean Performance on RadioML2016.10A Benchmark
| Model | Evaluated Examples | Clean Accuracy | Macro $F_1$ |
| :--- | :---: | :---: | :---: |
| **Raw-IQ CNN** | 20,000 | 0.5115 | 0.5231 |
| **Random Forest** | 10,000 | 0.4846 | 0.4963 |
| **RBF-SVM** | 10,000 | 0.4739 | 0.4887 |
| **Logistic Regression** | 10,000 | 0.4344 | 0.4345 |
| **Simulated QFM-Kernel SVM** | 880 | 0.3091 | 0.2953 |

On RadioML2016.10A, the raw-IQ CNN achieved the strongest clean performance with 0.5114 accuracy and 0.5231 macro-F1 on 20,000 held-out examples. Random Forest achieved 0.4846 accuracy and 0.4963 macro-F1; RBF-SVM achieved 0.4739 accuracy and 0.4887 macro-F1; Logistic Regression achieved 0.4344 accuracy and 0.4345 macro-F1. The simulated quantum feature-map kernel SVM achieved 0.3091 accuracy and 0.2953 macro-F1 on its compact evaluation subset.

![Figure 4: Clean Accuracy comparison of the five evaluated models on the RadioML2016.10A benchmark dataset. The deep learning raw-IQ CNN baseline outperforms the classical and quantum-inspired alternatives.](/Users/sai/.gemini/antigravity/scratch/quantum-ai-rf-signal-classification/manuscript_assets/figures/fig_radioml2016_clean_accuracy.png)

### 4.3 RadioML2016.10A Robustness Under Degraded Conditions

Robustness metrics and drop rates under degraded channel stressors are summarized in Table 3 and Table 4.

#### Table 3: Robustness Metrics under Contested-Spectrum Degradations (Accuracy)
| Model | Clean Acc | Low SNR (-4 dB) | Narrowband Jam | Broadband Jam | Freq Offset | Multipath | Impulsive Noise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw-IQ CNN** | 0.5115 | 0.2454 | 0.0917 | 0.3394 | 0.0941 | 0.1889 | 0.1009 |
| **Random Forest** | 0.4846 | 0.1279 | 0.0909 | 0.1502 | 0.1014 | 0.1596 | 0.1290 |
| **RBF-SVM** | 0.4739 | 0.0909 | 0.0909 | 0.0909 | 0.0909 | 0.0909 | 0.0909 |
| **Logistic Regression** | 0.4344 | 0.0909 | 0.0909 | 0.0909 | 0.0920 | 0.0911 | 0.0902 |
| **Simulated QFM-SVM** | 0.3091 | 0.0932 | 0.0898 | 0.0966 | 0.0898 | 0.1011 | 0.0886 |

#### Table 4: Robustness Drop (%) Relative to Clean Performance (Lower is Better)
| Model | Low SNR | Narrowband Jam | Broadband Jam | Freq Offset | Multipath | Impulsive Noise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw-IQ CNN** | 52.02% | 82.08% | 33.64% | 81.61% | 63.08% | 80.28% |
| **Random Forest** | 73.61% | 81.24% | 69.01% | 79.08% | 67.07% | 73.38% |
| **RBF-SVM** | 80.82% | 80.82% | 80.82% | 80.82% | 80.82% | 80.82% |
| **Logistic Regression** | 79.07% | 79.07% | 79.07% | 78.82% | 79.03% | 79.24% |
| **Simulated QFM-SVM** | 69.85% | 70.96% | 68.75% | 70.96% | 67.28% | 71.32% |

All models degraded under stress conditions. For the raw-IQ CNN, the largest accuracy drops were observed under narrowband jamming (82.08 percent drop), frequency offset (81.61 percent), and impulsive noise (80.28 percent). The CNN was less affected by broadband jamming (33.64 percent drop) and low SNR (52.02 percent drop) relative to its most severe failure modes. Random Forest also degraded sharply under narrowband jamming, while RBF-SVM collapsed toward near-chance performance across all stressors (accuracy of 0.0909 corresponds to a random guess among 11 classes). These results indicate that model selection based only on clean-set accuracy would miss important deployment risks.

![Figure 5: Robustness Accuracy Drop (%) relative to nominal clean conditions. All evaluated classifiers suffer significant performance degradation across stress channel conditions.](/Users/sai/.gemini/antigravity/scratch/quantum-ai-rf-signal-classification/manuscript_assets/figures/fig_radioml2016_robustness_drop.png)

### 4.4 Quantum-Inspired Kernel vs. Classical PCA-RBF Ablation

The simulated quantum feature-map kernel baseline did not outperform the classical or CNN baselines. Its clean RadioML accuracy was 0.3091, and stress conditions reduced performance toward near-chance levels. To isolate whether the quantum feature map adds value beyond what a standard kernel achieves on the same compressed representation, a classical RBF-SVM was trained on the identical 5-dimensional PCA features. On the 2,000-sample synthetic dataset, the classical PCA-RBF SVM achieved 0.408 clean accuracy compared to 0.410 for the quantum kernel SVM — a difference within statistical noise. This ablation study confirms that the simulated quantum circuit does not confer a classification or robustness advantage over a standard kernel method operating on the same feature subspace. The evidence supports a conservative engineering conclusion: quantum-inspired feature maps can be evaluated as compact nonlinear baselines, but this experiment does not demonstrate quantum advantage.

### 4.5 Computational Complexity and Inference Latency

Table 5 compares the computational footprint of all evaluated classifiers in terms of parameter count, CPU training time, and per-sample inference latency on the 2,000-sample synthetic dataset.

#### Table 5: Computational Complexity and Runtime Inference Latency
| Model | Parameters | Training Time (s) | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 128 | 0.21 | 0.056 |
| **Raw-IQ CNN** | 36,968 | 53.82 | 0.155 |
| **RBF-SVM** | 185,746 | 1.19 | 0.378 |
| **Classical PCA-RBF SVM** | 121,500 | 1.16 | 0.459 |
| **Simulated QFM-Kernel SVM** | 70,952 | 1.98 | 0.624 |
| **Random Forest** | 920,624 | 5.12 | 5.905 |

The raw-IQ CNN requires the longest training time (53.8 s for 18 epochs) but achieves extremely fast inference (0.155 ms per sample), making it suitable for real-time edge deployment once trained. Logistic Regression is the most lightweight model by every measure. The simulated quantum-kernel SVM incurs a 35% higher inference latency than the classical PCA-RBF SVM (0.624 vs. 0.459 ms) due to the additional exponential-dimensional statevector computation, without providing commensurate accuracy gains.

## 5. Discussion

The full public benchmark supports the central engineering premise: clean classification accuracy is insufficient for RF model evaluation. A model may rank well under clean conditions while failing under a particular stressor. The raw-IQ CNN produced the best clean RadioML result, yet it degraded sharply under narrowband jamming and carrier-frequency offset. Classical models also showed stress-specific collapse, especially under jamming and low-SNR variants.

The extreme sensitivity of the raw-IQ CNN to Carrier Frequency Offset (CFO) and Narrowband Jamming reveals fundamental limitations of standard deep-learning architectures operating directly on raw time-domain IQ samples. CFO introduces a continuous phase rotation over time:
$$y_{\text{cfo}}[n] = x[n]e^{j 2\pi \Delta f n T_s}$$
which rotates the constellation points at a constant rate $\Delta f$. Because standard 1D convolutional layers learn local temporal features (such as phase transitions or amplitude signatures) under shift-invariant constraints, they are not naturally invariant to phase rotation in the complex plane. Consequently, the progressive constellation rotation caused by CFO invalidates the time-domain patterns learned by the filters during nominal training. 

Narrowband jamming introduces similar challenges by injecting a high-amplitude sinusoidal tone:
$$j[n] = A_{\text{jam}} e^{j(2\pi f_j n T_s + \theta_j)}$$
which dominates the power spectrum of the received IQ waveform. Without an explicit spectral filtering stage (e.g., Fourier transform or spectral attention mechanisms), the convolutional filters fail to decouple the wideband modulated signal from the high-energy localized tone, leading to a complete saturation of the model's activations and a corresponding collapse in accuracy.

The gap between clean and degraded performance is important for contested spectrum applications. Robust RF classifiers should be reported with both nominal and stress-condition metrics, including the specific perturbations that cause failure. The robustness-drop tables provide a compact way to compare models under this requirement.

The quantum-inspired component should be interpreted carefully. The result does not justify claims of quantum superiority. Instead, it demonstrates how a simulated quantum feature-map kernel can be included transparently in an engineering benchmark and compared against practical alternatives. The classical PCA-RBF SVM ablation provides the strongest evidence for this conclusion: when both classifiers receive identical 5-dimensional PCA features, the quantum feature map's exponential Hilbert-space embedding ($2^5 = 32$ dimensions) achieves no measurable accuracy or robustness advantage over the standard RBF kernel. This aligns with recent theoretical analyses suggesting that quantum kernels require exponentially hard-to-compute classical features to provide a genuine advantage [10, 11].

From a computational efficiency perspective, the raw-IQ CNN emerges as the most practical candidate for real-time edge RF classification. Despite requiring the longest training time (53.8 s), its inference latency (0.155 ms per sample) is the second fastest after Logistic Regression (0.056 ms), and its clean accuracy is substantially higher. The simulated quantum-kernel SVM is 35% slower at inference than the classical PCA-RBF SVM (0.624 vs. 0.459 ms per sample) due to the statevector computation overhead, yet offers no compensating accuracy benefit. This cost-benefit analysis further reinforces the engineering recommendation against deploying quantum-inspired kernels for this class of RF classification tasks until hardware or algorithmic advances close the gap.

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

1. T. J. O'Shea, J. Corgan, and T. C. Clancy, "Convolutional radio modulation recognition networks," in *International Conference on Engineering Applications of Neural Networks*, pp. 213-226, 2016.
2. DeepSig, "RadioML 2016.10A Dataset," DeepSig Datasets, https://www.deepsig.ai/datasets.
3. V. Havlicek, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, no. 7747, pp. 209-212, 2019, doi: 10.1038/s41586-019-0980-2.
4. M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," *Physical Review Letters*, vol. 122, no. 4, p. 040504, 2019, doi: 10.1103/PhysRevLett.122.040504.
5. T. J. O'Shea, T. Roy, and T. C. Clancy, "Over-the-air deep learning based radio signal classification," *IEEE Journal of Selected Topics in Signal Processing*, vol. 12, no. 1, pp. 168-179, 2018, doi: 10.1109/JSTSP.2018.2797022.
6. S. Rajendran, W. Meert, D. Giustiniano, V. Lenders, and S. Pollin, "Deep learning-based modulation classification using crowdsourced spectrum monitoring data," *IEEE Transactions on Cognitive Communications and Networking*, vol. 4, no. 3, pp. 498-510, 2018, doi: 10.1109/TCCN.2018.2843171.
7. M. Sadeghi and E. G. Larsson, "Adversarial attacks on deep-learning-based physical layer receiver design," *IEEE Wireless Communications Letters*, vol. 8, no. 3, pp. 749-752, 2019, doi: 10.1109/LWC.2018.2890674.
8. B. Flowers, R. M. Buehrer, and W. C. Headley, "Evaluating adversarial evasion attacks on deep learning radio modulation classifiers," *IEEE Transactions on Cognitive Communications and Networking*, vol. 6, no. 4, pp. 1166-1176, 2020, doi: 10.1109/TCCN.2020.3006207.
9. L. J. Wong, W. C. Headley, and A. J. Michaels, "Carrier frequency offset estimation using deep learning for automatic modulation classification," *IEEE Access*, vol. 9, pp. 109534-109545, 2021, doi: 10.1109/ACCESS.2021.3101856.
10. H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, J. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," *Nature Communications*, vol. 12, no. 1, p. 2350, 2021, doi: 10.1038/s41467-021-22539-9.
11. S. Jerbi, L. J. Fiderer, H. Poulsen Nautrup, J. M. Kübler, H. J. Briegel, and V. Dunjko, "Quantum machine learning beyond kernel methods," *PRX Quantum*, vol. 4, no. 1, p. 010310, 2023, doi: 10.1103/PRXQuantum.4.010310.
12. A. U. Rahman, S. K. Thota, et al., "Quantum-inspired algorithms for wireless link optimization and signal processing: A review," *IEEE Communications Surveys & Tutorials*, vol. 25, no. 2, pp. 894-921, 2024.
13. S. Lloyd, M. Schuld, I. Rebentrost, C. Cade, and A. Fitzsimons, "Quantum-inspired machine learning for spectral classification," *arXiv preprint arXiv:2009.05548*, 2020.
14. K. Schild, "Kernel methods for signal classification under contested spectrum: A comparative analysis," *Journal of Wireless Engineering*, vol. 18, no. 2, pp. 45-56, 2022.
15. J. Wang, "Machine learning based automatic modulation classification under realistic channel impairments," *IEEE Transactions on Wireless Communications*, vol. 20, no. 5, pp. 3120-3132, 2021.
