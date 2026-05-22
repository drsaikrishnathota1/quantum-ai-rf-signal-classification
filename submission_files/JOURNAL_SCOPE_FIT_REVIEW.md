# Journal Scope-Fit Review

Target journal: Results in Engineering

Purpose: local quality-control note before submission. Do not upload unless the
journal asks for a response or supporting explanation.

## Scope Signals Checked

- Results in Engineering accepts technically correct engineering papers across
  electronics engineering, electrical engineering, computers, and AI/ML
  engineering.
- The journal explicitly warns that AI/ML papers need novelty beyond applying a
  known AI algorithm to an existing problem.
- A directly relevant Results in Engineering AMR paper used RadioML2016.10A and
  RadioML2016.10B, reported accuracy, model efficiency, low-SNR robustness, and
  ablation evidence.
- Recent Results in Engineering AI papers commonly state an applied engineering
  problem, define the proposed method or protocol, include empirical evidence,
  and report practical metrics such as runtime, robustness, uncertainty, or
  engineering cost.

## Recent Journal-Style Examples Considered

1. Improved automatic modulation recognition using deep learning with additive
   attention, Results in Engineering, 2025, doi: 10.1016/j.rineng.2025.104783.
2. Physics-informed neural networks for 3D transient thermal simulation of MIG
   welding, Results in Engineering, 2026, doi: 10.1016/j.rineng.2026.109549.
3. Leveraging artificial intelligence models for efficient chromium reduction,
   Results in Engineering, 2025, doi: 10.1016/j.rineng.2025.104599.
4. Smart material estimation for the engineering, procurement, and construction
   sector, Results in Engineering, 2025, doi: 10.1016/j.rineng.2025.105802.
5. Decoupled deep learning for geohazards mapping in oceanic deep drilling,
   Results in Engineering, 2025, doi: 10.1016/j.rineng.2025.108386.
6. Machine learning for CO2 geological storage and CO2-enhanced oil recovery in
   hydrocarbon reservoirs, Results in Engineering, 2026,
   doi: 10.1016/j.rineng.2026.110928.

## Fit Decision

The manuscript should be submitted as an Original Article only if it is framed
as RSC-Bench, a reproducible robustness-evaluation protocol for RF signal
classification. It should not be framed as a generic comparison of existing
classifiers or as a quantum-advantage paper.

## Risk Controls Added

- Named protocol: RSC-Bench.
- Precise novelty sentence in the abstract and introduction.
- Formal pseudocode and computational complexity analysis.
- Expanded reference list above 30 references.
- Public RadioML2016.10A benchmark validation.
- Same-feature PCA-RBF SVM ablation against the simulated quantum kernel.
- AI-assisted writing disclosure and unresolved-field checks.
