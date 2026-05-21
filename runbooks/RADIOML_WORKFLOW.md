# RadioML Public Benchmark Workflow

This is the next major evidence step for the Results in Engineering paper.

## 1. Download Dataset Manually

Download RadioML2016.10A from the original provider and place the pickle file
under:

```text
data/radioml/RML2016.10a_dict.pkl
```

Do not commit this file. It is intentionally ignored by Git.

## 2. Convert To Project NPZ Format

```bash
.venv/bin/python scripts/prepare_radioml2016_npz.py \
  --input data/radioml/RML2016.10a_dict.pkl \
  --out data/radioml/radioml2016_10a_clean.npz
```

For a faster smoke test:

```bash
.venv/bin/python scripts/prepare_radioml2016_npz.py \
  --input data/radioml/RML2016.10a_dict.pkl \
  --out data/radioml/radioml2016_10a_clean_smoke.npz \
  --max-examples-per-mod-snr 100
```

## 3. Add Stress Conditions

```bash
.venv/bin/python scripts/add_stress_conditions_to_npz.py \
  --input data/radioml/radioml2016_10a_clean.npz \
  --out data/radioml/radioml2016_10a_stress.npz
```

## 4. Train Classical Baselines

```bash
.venv/bin/python scripts/train_pilot_classifiers.py \
  --data data/radioml/radioml2016_10a_stress.npz \
  --out results/radioml2016_classical
```

## 5. Train Raw-IQ CNN

```bash
.venv/bin/python scripts/train_cnn_iq_baseline.py \
  --data data/radioml/radioml2016_10a_stress.npz \
  --out results/radioml2016_cnn \
  --epochs 40 \
  --batch-size 512
```

## 6. Run Quantum-Inspired Kernel Baseline

```bash
.venv/bin/python scripts/train_quantum_inspired_kernel.py \
  --data data/radioml/radioml2016_10a_stress.npz \
  --out results/radioml2016_quantum_kernel \
  --qubits 5 \
  --max-train-per-class 120 \
  --max-test-per-class 80
```

## 7. Manuscript Use

The final paper should report synthetic and RadioML evidence separately:

- Synthetic controlled protocol for engineering stress testing
- RadioML public benchmark for external reproducibility

This separation is important for reviewer confidence.
