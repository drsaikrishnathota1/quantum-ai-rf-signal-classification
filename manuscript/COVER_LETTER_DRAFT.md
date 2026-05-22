# Cover Letter Draft

Dear Editor,

Please consider the manuscript titled "Robustness Benchmarking of Classical,
Deep, and Quantum-Inspired AI for RF Signal Classification in Contested
Spectrum Environments" for publication as an Original Article in Results in
Engineering.

The manuscript addresses an engineering problem in robust RF signal
classification: AI systems that perform adequately under nominal signal
conditions can degrade sharply under low SNR, jamming, multipath, frequency
offset, and impulsive interference. The study develops a reproducible
evaluation workflow that compares classical machine-learning baselines, a
raw-IQ convolutional neural network, a simulated quantum feature-map kernel
classifier, and a same-feature PCA-RBF SVM ablation under controlled
degraded-spectrum conditions. The final evidence package includes a full
RadioML2016.10A GPU validation with 220,000 converted examples, a scaled
synthetic benchmark, clean benchmark results, robustness-drop analysis,
computational complexity reporting, and generated manuscript figures and
tables.

The manuscript has also been strengthened with an expanded literature survey
covering classical AMC, deep-learning RF classification, robust/adversarial
RFML, jamming, and quantum-kernel learning. The novelty is positioned as a
reproducible robustness-benchmarking and ablation workflow, not as a simple
application of an existing classifier.

The work is framed cautiously. It does not claim quantum advantage. Instead, it
uses the quantum-inspired component as an evaluated compact feature-map
baseline and compares it transparently against strong classical and
deep-learning alternatives. In particular, the same-feature PCA-RBF ablation
shows that the simulated quantum kernel does not provide measurable advantage
over a classical RBF kernel in this configuration. This framing is intended to
support engineering reproducibility and reviewer confidence.

The manuscript, code, result tables, and generated figures will be made
available through the accompanying repository. The RadioML2016.10A raw dataset
is not redistributed and is referenced under its original access and license
terms.

Sincerely,

Dr. Sai Krishna Thota  
Independent Researcher, USA  
drsaikrishnathota@ieee.org  
https://orcid.org/0009-0008-5246-9421
