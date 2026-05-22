# Manual Submission Script

Target journal:

**Results in Engineering**  
Homepage: https://www.sciencedirect.com/journal/results-in-engineering

Use this file for copy-paste fields during Elsevier Editorial Manager submission.
This version is aligned to the completed synthetic and RadioML2016.10A full GPU
benchmark evidence package.

## Article Type

Original Article

## Full Title

RSC-Bench: Robustness Benchmarking of Classical, Deep, and Quantum-Inspired AI for RF Signal Classification in Contested Spectrum Environments

## Short Title

RSC-Bench for RF Signal Classification

## Author Details

Dr. Sai Krishna Thota  
Independent Researcher, USA  
Email: drsaikrishnathota@ieee.org  
ORCID: https://orcid.org/0009-0008-5246-9421

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

## Keywords

automatic modulation classification; RF signal classification; quantum-inspired machine learning; quantum kernel; robust artificial intelligence; contested spectrum; jamming; low SNR; deep learning; cognitive radio

## Additional Comments To Publication Office

Dear Editorial Office,

Please consider this manuscript as an Original Article for Results in
Engineering. The work presents RSC-Bench, a reproducible engineering protocol for
robustness-aware RF signal classification under degraded and contested-spectrum
conditions. The study compares classical machine learning, a raw-IQ deep
learning baseline, a simulated quantum feature-map kernel baseline, and a
same-feature PCA-RBF SVM ablation using transparent held-out evaluation,
robustness-drop analysis, formal pseudocode, computational complexity reporting, and
reproducibility artifacts. The manuscript is explicitly framed as a robustness
benchmark and ablation study, not as a quantum-advantage claim.

The manuscript is submitted as independent research by Dr. Sai Krishna Thota.
All data-generation scripts, evaluation code, summary tables, and figure
generation scripts will be made available in the accompanying repository. The
RadioML2016.10A raw dataset is not redistributed and is referenced according to
its original access and license terms.

Sincerely,  
Dr. Sai Krishna Thota

## Funding Statement

This research received no external funding.

## Conflict Of Interest Statement

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

## Ethics Statement

This study uses synthetic RF/IQ simulations and the public RadioML2016.10A RF
benchmark dataset. No human participants, animal subjects, patient records, or
personally identifiable information are involved. Formal ethics approval was
therefore not required.

## Data And Code Availability Draft

The code, configuration files, result tables, and generated manuscript figures
will be available at:

https://github.com/drsaikrishnathota1/quantum-ai-rf-signal-classification

Synthetic datasets are generated by the repository scripts and are not tracked
in Git because they are reproducible binary artifacts. The RadioML2016.10A raw
dataset is not redistributed; access is described through the original dataset
provider or validated public mirror and license terms.

## AI-Assisted Writing Disclosure

AI-assisted drafting and code-generation tools were used for editorial
drafting, code scaffolding, and consistency checking. The author reviewed,
executed, and verified the analyses, metrics, and final text.
