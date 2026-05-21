# Next Steps

## Where We Are Now

The project has started successfully. We now have:

- journal target
- paper blueprint
- RunPod plan
- synthetic RF/IQ dataset generator
- pilot classical classifiers
- pilot robustness evaluation
- first local result tables and figures

## Next Step 1: Scale Synthetic Pilot

Run a larger synthetic dataset locally or on RunPod:

```bash
.venv/bin/python scripts/generate_synthetic_iq_dataset.py --samples-per-class 1000 --out data/synthetic_iq_1k.npz
.venv/bin/python scripts/train_pilot_classifiers.py --data data/synthetic_iq_1k.npz --out results/synthetic_1k
.venv/bin/python scripts/make_pilot_figures.py --metrics results/synthetic_1k/pilot_metrics.csv --drops results/synthetic_1k/pilot_robustness_drop.csv --out results/synthetic_1k/figures
```

Expected local runtime:

- 5-20 minutes depending on Mac load.

## Next Step 2: Add CNN Baseline

Create a PyTorch 1D CNN trained on raw IQ samples.

Outputs:

- clean accuracy
- accuracy by SNR
- robustness under jamming/noise/frequency-offset/multipath
- confusion matrix

## Next Step 3: Add Quantum-Inspired Baseline

Start with a compact feature vector from the classical script.

Quantum-inspired candidates:

- RBF-SVM as classical kernel baseline
- quantum-kernel SVM using Qiskit Aer simulator
- variational classifier only on a small reduced subset if simulation cost is high

Important wording:

The paper should say "quantum-inspired" or "simulated quantum-kernel baseline."
Do not claim quantum advantage.

## Next Step 4: Public Benchmark

Add RadioML2016.10A first.

We need to download the dataset from DeepSig. If the download requires a login
or form, Dr. Sai must complete that step, then place the dataset in:

```text
data/radioml/
```

## Next Step 5: Paper Assets

After full experiments:

- Table 1: dataset/simulation settings
- Table 2: model settings
- Table 3: clean benchmark comparison
- Table 4: robustness degradation comparison
- Table 5: ablation study
- Figure 1: workflow
- Figure 2: IQ constellation examples
- Figure 3: accuracy vs SNR
- Figure 4: robustness drop
- Figure 5: confusion matrix

