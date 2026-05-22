# Research Blueprint

## Target Journal

**Results in Engineering**  
Publisher: Elsevier / ScienceDirect  
Homepage: https://www.sciencedirect.com/journal/results-in-engineering

Reason for fit:

- The journal includes AI and ML engineering.
- The paper has a clear engineering problem: robust RF signal classification.
- The work can be validated using public/simulated data and reproducible code.
- The journal's page emphasizes technical correctness, engineering grounding,
  and novelty beyond re-applying a known AI method.

## Working Title

**Robustness Benchmarking of Classical, Deep, and Quantum-Inspired AI for RF Signal Classification in Contested Spectrum Environments**

## Positioning

The safest journal framing is a robustness-benchmarking and ablation paper,
not a quantum-advantage paper. The contribution is the reproducible comparison
of classical ML, raw-IQ CNN, simulated quantum-kernel SVM, and same-feature
PCA-RBF SVM under clean and degraded RF conditions.

## Main Research Question

How do classical AI, deep learning, and quantum-inspired feature-map classifiers
compare when RF signal classification is evaluated under realistic degradation
conditions such as low SNR, jamming, multipath fading, carrier frequency offset,
and impulsive interference?

## Engineering Problem

Automatic modulation classification is useful for spectrum monitoring, secure
communications, cognitive radio, UAV communications, radar/RF surveillance, and
defense-adjacent electronic support systems. A model that performs well on clean
or idealized RF signals may fail when the channel is contested or degraded.

## Proposed Contributions

1. A reproducible RF signal classification workflow using simulated IQ signals
   and public RadioML data.
2. A controlled contested-spectrum robustness protocol:
   - low SNR
   - narrowband jamming
   - broadband jamming
   - multipath fading
   - carrier frequency offset
   - impulsive noise
3. Baseline comparison:
   - classical ML on engineered RF features
   - raw-IQ CNN on signal samples
   - simulated quantum-kernel classifier on compact RF features
   - same-feature PCA-RBF SVM ablation
4. Robustness-drop analysis by degradation type and modulation class.
5. Reproducibility package with scripts, config files, result logs, figures, and
   model metadata.

## Datasets

### Phase 1: Synthetic Pilot

Generate IQ waveforms for:

- BPSK
- QPSK
- 8PSK
- QAM16
- QAM64
- BFSK
- CPFSK-like tone shift
- AM-DSB
- FM-like signal

Synthetic pilot value:

- fast
- fully reproducible
- no license issues
- lets us test the entire experiment pipeline

### Phase 2: Public Benchmark

Use DeepSig RadioML datasets:

- RML2016.10A for fast baseline and debugging
- RML2018.01A for stronger paper-grade benchmark if storage/compute is available

Important license note:

DeepSig states that its public datasets are under CC BY-NC-SA 4.0. We will not
redistribute dataset files. We will cite the dataset source and provide download
instructions.

## Models

### Classical Baselines

- Logistic Regression
- Random Forest
- RBF-SVM
- XGBoost or LightGBM if available

Feature candidates:

- mean and variance of I/Q
- signal power
- amplitude statistics
- phase statistics
- spectral centroid
- spectral bandwidth
- fourth-order moments
- cumulant-inspired features

### Deep-Learning Baselines

- 1D CNN
- CNN-LSTM or CNN-GRU
- Temporal Convolutional Network
- Lightweight Transformer encoder

### Quantum-Inspired / Quantum ML Baselines

- Quantum-kernel SVM using compact standardized RF features
- Variational quantum classifier on a reduced feature vector
- Hybrid CNN feature extractor plus quantum-inspired classifier

Paper wording guardrail:

Use "quantum-inspired" or "NISQ-simulated quantum kernel." Do not claim quantum
advantage unless we have clear evidence.

## Robustness Protocol

For each model:

1. Train on clean + standard AWGN range.
2. Evaluate clean test set.
3. Evaluate low-SNR test sets.
4. Evaluate stress-test sets:
   - narrowband jammer
   - broadband jammer
   - multipath fading
   - carrier frequency offset
   - impulsive noise
5. Compute:
   - accuracy
   - macro F1
   - per-class F1
   - confusion matrix
   - robustness drop from clean
   - inference time

## Minimum Paper-Grade Tables

1. Dataset and simulation parameters
2. Model architecture and training settings
3. Clean benchmark performance
4. Accuracy by SNR
5. Robustness under contested-spectrum degradations
6. Ablation study for quantum-inspired component
7. Runtime and model complexity

## Minimum Paper-Grade Figures

1. End-to-end workflow diagram
2. Example IQ constellation plots
3. Accuracy vs SNR curve
4. Robustness drop bar chart
5. Confusion matrix under clean and jammed conditions
6. Quantum feature-map / hybrid architecture schematic

## Risk Management

Potential reviewer concern: "Quantum part is superficial."

Mitigation:

- Use compact feature maps with clear mathematical formulation.
- Compare with strong classical baselines.
- Present the quantum module as an evaluated engineering component, not magic.

Potential reviewer concern: "Defense claim is too broad."

Mitigation:

- Use "contested spectrum," "spectrum monitoring," "RF surveillance," and
  "defense-adjacent" framing.
- Avoid classified or operational claims.

Potential reviewer concern: "Only simulated data."

Mitigation:

- Add RadioML benchmark data after the synthetic pilot.
- Clearly separate synthetic and public-benchmark evidence.
