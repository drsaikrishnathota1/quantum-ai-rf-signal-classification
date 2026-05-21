# Quantum-AI RF Signal Classification Paper

Working title:

**Quantum-Inspired Hybrid AI for Robust RF Signal Classification in Contested Spectrum Environments**

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

> Compact quantum-inspired feature maps and quantum-kernel style classifiers can
> be evaluated as robustness-aware modules inside an RF signal classification
> workflow, and compared transparently against classical ML and deep-learning
> baselines under contested-spectrum degradations.

## Evidence Path

1. Start with a fully controlled synthetic RF simulation pilot.
2. Add public RadioML data for external benchmark evidence.
3. Train classical ML and deep-learning baselines.
4. Add quantum-inspired / quantum-kernel baselines on compact features.
5. Run robustness tests across low SNR and contested-spectrum degradations.
6. Generate paper tables, figures, reproducibility package, and final manuscript.

## First Milestone

Run the local pilot:

```bash
python3 scripts/generate_synthetic_iq_dataset.py --samples-per-class 300 --out data/pilot_iq.npz
python3 scripts/train_pilot_classifiers.py --data data/pilot_iq.npz --out results/pilot
```

The pilot is intentionally small. It proves the end-to-end flow before using
paid GPU resources.

