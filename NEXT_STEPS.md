# Next Steps

## Where We Are Now

The project has started successfully. We now have:

- journal target
- paper blueprint
- RunPod plan
- synthetic RF/IQ dataset generator
- pilot classical classifiers
- raw-IQ CNN baseline
- simulated quantum feature-map kernel baseline
- pilot robustness evaluation
- scaled synthetic result tables and figures
- draft submission/manual files
- draft manuscript sections

## Next Step 1: Public Benchmark Validation

Download RadioML2016.10A from the original provider and place it under:

```text
data/radioml/
```

Do not commit or redistribute the dataset. We should cite the original source
and license terms in the manuscript.

## Next Step 2: Paper-Grade Synthetic Scale

On RunPod or another GPU server:

```bash
python scripts/run_synthetic_pipeline.py \
  --samples-per-class 2000 \
  --data data/synthetic_iq_2k.npz \
  --classical-out results/synthetic_2k \
  --cnn-out results/cnn_2k \
  --quantum-out results/quantum_kernel_2k \
  --cnn-epochs 40 \
  --cnn-batch-size 512
```

## Next Step 3: Improve Quantum-Inspired Evidence

Current quantum-inspired evidence is useful but weaker than the classical and
CNN baselines. Next options:

- compare multiple feature-map depths
- compare qubit counts from 4 to 8
- add a classical-kernel ablation using the same PCA-reduced features
- optionally add Qiskit Aer/QSVC if runtime is acceptable

Important wording:

The paper should say "quantum-inspired" or "simulated quantum-kernel baseline."
Do not claim quantum advantage.

## Next Step 4: Final Paper Assets

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
