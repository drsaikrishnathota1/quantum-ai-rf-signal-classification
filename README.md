# Quantum-AI RF Signal Classification Paper

Working title:

**Robustness Benchmarking of Classical, Deep, and Quantum-Inspired AI for RF Signal Classification in Contested Spectrum Environments**

Target journal:

**Results in Engineering**  
ScienceDirect / Elsevier: https://www.sciencedirect.com/journal/results-in-engineering

## Purpose

This project builds a submission-grade research package for robust RF signal
classification using applied AI and quantum-inspired machine learning. The
paper will focus on an engineering problem: recognizing RF signal modulation
types when the spectrum is degraded by low SNR, jamming, multipath, frequency
offset, and impulsive noise.

## Core Claim

The paper will not claim quantum advantage. The safer and more defensible
claim is:

> Clean RF classification accuracy is not enough for contested-spectrum model
> selection; classical ML, raw-IQ CNN, simulated quantum-kernel, and same-feature
> PCA-RBF ablation baselines should be compared under reproducible degraded RF
> stress conditions.

## Evidence Path

1. Start with a fully controlled synthetic RF simulation pilot.
2. Add public RadioML data for external benchmark evidence.
3. Train classical ML and deep-learning baselines.
4. Add quantum-inspired / quantum-kernel baselines on compact features.
5. Run robustness tests across low SNR and contested-spectrum degradations.
6. Generate paper tables, figures, reproducibility package, and final manuscript.

## First Milestone

Run the scaled synthetic pipeline:

```bash
.venv/bin/python scripts/run_synthetic_pipeline.py --samples-per-class 2000 --cnn-epochs 40
```

The pipeline generates data, trains classical baselines, trains the raw-IQ CNN,
runs the simulated quantum feature-map kernel, and exports manuscript-ready
tables and figures.

## Current Evidence Snapshot

The current scaled synthetic run uses 16,000 examples across eight modulation
classes. Clean held-out accuracy is:

- Raw-IQ CNN: 0.654
- Random Forest: 0.592
- RBF-SVM: 0.570
- Logistic Regression: 0.554
- Simulated QFM-Kernel SVM: 0.410
- Classical PCA-RBF SVM: 0.408

The full public benchmark run on RadioML2016.10A is also complete. The raw-IQ
CNN achieved 0.5114 clean accuracy and 0.5231 macro-F1 on 20,000 held-out
examples, with severe degradation under narrowband jamming, frequency offset,
low SNR, multipath, and impulsive noise.

## Public Benchmark Workflow

After manually downloading RadioML2016.10A, follow:

```text
runbooks/RADIOML_WORKFLOW.md
```
