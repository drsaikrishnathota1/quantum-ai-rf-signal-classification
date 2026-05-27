# RSC-Bench: A Robustness Benchmark for RF Signal Classification — Evaluating Classical, Deep Learning, and Quantum-Inspired Methods Under Contested Spectrum Conditions

## Abstract

Automatic modulation classification is important for spectrum monitoring,
secure wireless communications, cognitive radio, unmanned-system links, and
defense-adjacent RF surveillance. However, many AI-based RF classifiers are
reported mainly under nominal channel assumptions, while practical receivers
must operate under low signal-to-noise ratio, narrowband and broadband
jamming, carrier-frequency offset, multipath, and impulsive interference. This
study presents RSC-Bench, a reproducible robustness-benchmarking protocol that
evaluates classical machine-learning baselines, a compact raw-IQ convolutional
neural network, a simulated quantum feature-map kernel classifier, and a
classical PCA-RBF SVM ablation baseline under seven contested-spectrum
conditions. The protocol is validated on a controlled synthetic IQ dataset with
16,000 examples across eight modulation families, and on the public
RadioML2016.10A benchmark containing 220,000 examples, 11 modulation classes,
and SNR values from -20 to 18 dB.

On the synthetic dataset the raw-IQ CNN achieved the highest clean accuracy
(0.654), followed by Random Forest (0.592), RBF-SVM (0.570), Logistic
Regression (0.554), the simulated quantum kernel SVM (0.410), and the PCA-RBF
ablation (0.408). On the full RadioML2016.10A GPU benchmark the raw-IQ CNN
achieved the strongest clean accuracy (0.559) and macro-F1 (0.574). Robustness
testing revealed severe degradation under narrowband jamming, carrier-frequency
offset, low SNR, multipath, and impulsive noise across all models, demonstrating
that clean accuracy alone is insufficient for RF model selection. A systematic
data-scarcity experiment across training set sizes of 5 to 200 samples per
class, repeated over five independent random seeds, found that the
quantum-inspired kernel underperforms classical baselines at all evaluated
training volumes. The classical PCA-RBF SVM and simulated quantum-kernel SVM
achieved statistically indistinguishable clean accuracy on the synthetic
dataset (0.408 vs. 0.410), confirming that no quantum-feature-map advantage is
supported by the evidence. The main engineering contribution is a transparent,
reproducible robustness protocol with full public-benchmark validation,
stress-condition failure analysis, low-data-regime characterization of the
quantum-inspired baseline, and computational complexity reporting.

Keywords: automatic modulation classification; RF signal classification;
quantum-inspired machine learning; quantum kernel; robust artificial
intelligence; contested spectrum; jamming; low SNR; deep learning; cognitive
radio

---

## 1. Introduction

[NO CHANGES — keep existing Introduction text exactly as-is]

---

## 2. Related Work And Research Gap

[NO CHANGES — keep existing Related Work text exactly as-is]

---

## 3. Contributions

This study contributes:

1. RSC-Bench: a reproducible RF robustness evaluation protocol covering clean,
   low-SNR, narrowband-jam, broadband-jam, frequency-offset, multipath, and
   impulsive-noise conditions on both synthetic IQ data and the public
   RadioML2016.10A benchmark.
2. A transparent comparison of engineered-feature classical ML (Logistic
   Regression, Random Forest, RBF-SVM), raw-IQ CNN, simulated quantum
   feature-map kernel SVM, and same-feature PCA-RBF SVM ablation baselines
   under matched evaluation conditions.
3. Full GPU benchmark results on RadioML2016.10A with 55,000 held-out examples,
   clean performance (CNN: 0.559 accuracy, 0.574 macro-F1), stress-condition
   metrics, and robustness-drop tables.
4. A data-scarcity characterization experiment evaluating all five model
   families across training set sizes of 5 to 200 samples per class with five
   independent random seeds, providing evidence on whether the quantum-inspired
   kernel is competitive in label-scarce regimes.
5. A direct ablation test of the quantum-inspired feature map against a
   classical RBF kernel on the same five-dimensional PCA representation,
   showing no measurable benefit.
6. A computational complexity and inference-latency analysis comparing
   parameter counts and per-sample execution time.
7. A cautious interpretation of the quantum-inspired component that avoids
   unsupported quantum-advantage claims and provides a replicable negative
   benchmark result for the community.

---

## 4. Materials And Methods

[NO CHANGES to sections 4.1–4.6 — keep existing text exactly as-is]

### 4.7 Data-Scarcity Evaluation Protocol

To characterize whether the quantum-inspired kernel offers competitive
performance in label-scarce regimes — a theoretical motivation for
quantum-feature-map methods — an additional experiment was conducted sweeping
training set sizes of 5, 10, 20, 50, 100, and 200 samples per class. For each
training set size, five independent random seeds were used to sample training
examples from the synthetic IQ dataset, with a fixed held-out test pool of 60
examples per class shared across all conditions. All five model families
(Logistic Regression, Random Forest, RBF-SVM, raw-IQ CNN, and simulated
QFM-Kernel SVM) were evaluated at each training volume. Mean accuracy and
standard deviation across five seeds are reported. This protocol enables
direct comparison of model families under severe data scarcity and provides
statistically grounded evidence on whether the quantum-inspired kernel closes
the performance gap at low training volumes.

### 4.8 Evaluation Protocol

[Keep existing Section 4.7 text here, renumbered to 4.8]

---

## 5. Results

### 5.1 Synthetic dataset results

[NO CHANGES — keep existing Section 5.1 text exactly as-is]

---

### 5.2 RadioML2016.10A clean performance

The full GPU RadioML benchmark is the primary public-dataset evidence for this
study. The clean classification table is provided in:

`manuscript_assets/tables/table_radioml2016_full_clean_performance.csv`

On RadioML2016.10A with 55,000 held-out examples, the raw-IQ CNN achieved the
strongest clean performance with 0.559 accuracy and 0.574 macro-F1, restored
from the best epoch (epoch 23) of a 40-epoch GPU training run. Random Forest
achieved 0.497 accuracy and 0.508 macro-F1; RBF-SVM achieved 0.488 accuracy
and 0.502 macro-F1; Logistic Regression achieved 0.448 accuracy and 0.448
macro-F1. The simulated quantum feature-map kernel SVM achieved 0.309 accuracy
and 0.295 macro-F1 on its compact evaluation subset (120 train per class, 80
test per class).

---

### 5.3 RadioML2016.10A robustness under degraded conditions

Robustness metrics and drop rates under degraded channel stressors are
provided in:

`manuscript_assets/tables/table_radioml2016_full_robustness_metrics.csv`
`manuscript_assets/tables/table_radioml2016_full_robustness_drop.csv`

All models degraded substantially under stress conditions. For the raw-IQ CNN,
the largest accuracy drops were observed under narrowband jamming (83.7 percent
drop to 0.091 accuracy), frequency offset (79.2 percent drop to 0.116
accuracy), and impulsive noise (81.5 percent drop to 0.103 accuracy). The CNN
was comparatively more resilient to broadband jamming (40.2 percent drop to
0.334 accuracy) and multipath (69.9 percent drop to 0.168 accuracy). Random
Forest degraded sharply under narrowband jamming, while RBF-SVM collapsed
toward near-chance performance across all stressors. Logistic Regression
similarly degraded to near-chance levels under all stress conditions. These
results confirm that model selection based only on clean-set accuracy would
miss critical deployment risks in contested spectrum environments.

---

### 5.4 Quantum-inspired kernel versus PCA-RBF ablation

[NO CHANGES — keep existing Section 5.4 text exactly as-is]

---

### 5.5 Data-scarcity experiment: quantum-inspired kernel in the low-data regime

To test whether the simulated quantum feature-map kernel offers competitive
performance when labeled training data is severely limited, all five model
families were evaluated across training set sizes ranging from 5 to 200
samples per class, repeated over five independent random seeds. Results are
summarized in Table X (manuscript_assets/tables/table_scarcity_accuracy_vs_trainsize.csv)
and illustrated in Figures Y and Z (manuscript_assets/figures/fig_scarcity_accuracy_vs_trainsize.png
and fig_scarcity_crossover_zoom.png).

The QFM-Kernel SVM underperformed classical baselines at every evaluated
training set size. Mean accuracy across five seeds at 5 samples per class was
0.127 for QFM-Kernel SVM versus 0.150 for Random Forest and 0.173 for RBF-SVM.
At 200 samples per class the gap widened: QFM-Kernel SVM achieved 0.160 mean
accuracy versus 0.330 for Random Forest and 0.334 for RBF-SVM. No crossover
was observed at any evaluated training volume, and the performance gap between
the quantum-inspired method and the best classical baseline widened
monotonically as training data increased.

These results indicate that the ZZ-feature-map encoding applied to
five-dimensional PCA-reduced IQ features does not capture discriminative
structure that classical kernels cannot, even in severely data-limited
conditions. This is itself a useful negative benchmark result: it provides
quantitative evidence that practitioners considering quantum-inspired feature
maps for spectrum classification should not expect a data-efficiency advantage
with this architecture on this problem class.

---

### 5.6 Computational complexity and inference latency

[Keep existing Section 5.5 text here, renumbered to 5.6]

---

## 6. Discussion

The full public benchmark supports the central engineering premise: clean
classification accuracy is insufficient for RF model evaluation in contested
spectrum environments. The raw-IQ CNN produced the best clean RadioML result
(0.559 accuracy on 55,000 held-out examples), yet degraded sharply under
narrowband jamming and carrier-frequency offset. Classical models also showed
stress-specific collapse, especially under jamming and low-SNR variants.

The data-scarcity experiment adds an important dimension to the quantum-inspired
component analysis. Quantum kernel methods are theoretically motivated for
label-scarce settings because high-dimensional Hilbert-space embeddings may
separate data with fewer examples. However, the experimental evidence across
five seeds and six training set sizes does not support this hypothesis for the
ZZ-feature-map kernel on PCA-reduced IQ features: the classical RBF-SVM
outperformed it at every training volume from 5 to 200 samples per class.
This negative result is scientifically informative — it constrains the
practical operating regime of this quantum-inspired architecture and provides
a reproducible baseline for future work.

[Keep remaining Discussion paragraphs exactly as-is]

---

## 7. Limitations

This study is simulation and public-benchmark based. It does not use live RF
captures, hardware-in-the-loop evaluation, over-the-air adversarial testing, or
classified defense data. RadioML2016.10A is a useful public benchmark, but it
does not represent every operational RF channel or emitter environment. The
stress transformations are controlled and reproducible rather than exhaustive.
The simulated quantum feature-map kernel is evaluated on a compact subset
because kernel methods scale poorly with the number of training examples. The
data-scarcity experiment uses the synthetic dataset only; results may differ on
RadioML2016.10A or real-world captures. No hardware quantum processor was used;
all quantum-inspired results are classical simulations of quantum feature maps.

The paper should therefore be interpreted as a reproducible engineering
benchmark and robustness audit, not as a claim of operational deployment
readiness or quantum advantage.

---

## 8. Conclusion

This work establishes RSC-Bench, a reproducible evaluation protocol for
robustness-aware RF signal classification under degraded spectrum conditions.
The full RadioML2016.10A GPU benchmark — 220,000 examples, 40 training epochs,
55,000 held-out examples — shows that the raw-IQ CNN achieves the strongest
clean performance (0.559 accuracy) among the tested models, while also
revealing large stress-condition failure modes under narrowband jamming,
frequency offset, and impulsive noise. A systematic data-scarcity experiment
across five random seeds and six training set sizes confirms that the simulated
quantum feature-map kernel underperforms classical baselines at all evaluated
training volumes, providing a quantitative negative benchmark result for
quantum-inspired methods on this problem class. Classical models and the
simulated quantum-inspired kernel provide useful comparators, while the
quantum-inspired result is best framed as a transparent, reproducible baseline
rather than a competitive method. The main engineering contribution is a
reproducible robustness protocol, full GPU benchmark evidence, and a
statistically rigorous characterization of quantum-inspired feature maps for
AI-based RF classification in contested-spectrum environments.

---

## 9. Submission Integrity And AI-Assisted Writing Disclosure

[NO CHANGES — keep existing Section 9 text exactly as-is]

---

## References

[NO CHANGES — keep existing References exactly as-is]

---

## EDITOR NOTES (remove before submission)

Key changes from previous draft:
- Title updated: removed "hybrid AI", changed to "Methods"
- Abstract updated with new GPU numbers: CNN 0.559 accuracy (was 0.511)
- Section 3 Contributions: added C4 (data scarcity) and C7 (negative result)
- Section 4.7 added: Data-Scarcity Evaluation Protocol (new)
- Section 5.2 updated: new RadioML numbers from full GPU run
- Section 5.3 updated: new robustness drop numbers
- Section 5.5 added: Data-scarcity experiment results (new)
- Section 6 Discussion: added data-scarcity paragraph
- Section 7 Limitations: added data-scarcity and hardware limitations
- Section 8 Conclusion: updated with new numbers and scarcity finding
