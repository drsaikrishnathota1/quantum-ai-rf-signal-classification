# Full Experiment Workflow

This runbook keeps the work reproducible from local Mac to RunPod.

## Local Quick Validation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_synthetic_pipeline.py --samples-per-class 500 --cnn-epochs 18
```

Expected output:

- `results/synthetic_500/`
- `results/cnn_500/`
- `results/quantum_kernel_500/`
- `manuscript_assets/tables/`
- `manuscript_assets/figures/`
- `ANALYSIS_REPORT_SYNTHETIC_500.md`

## RunPod Paper-Grade Synthetic Run

Use one H100 or A100 GPU. One GPU is enough.

```bash
cd /workspace
git clone https://github.com/drsaikrishnathota1/quantum-ai-rf-signal-classification.git
cd quantum-ai-rf-signal-classification
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_synthetic_pipeline.py \
  --samples-per-class 2000 \
  --data data/synthetic_iq_2k.npz \
  --classical-out results/synthetic_2k \
  --cnn-out results/cnn_2k \
  --quantum-out results/quantum_kernel_2k \
  --cnn-epochs 40 \
  --cnn-batch-size 512
```

## Public Benchmark Step

Download RadioML2016.10A manually from the original provider, then place it
under:

```text
data/radioml/
```

Do not commit or redistribute the dataset files.

## Evidence Needed For Final Paper

- Synthetic results with at least 2,000 samples per class
- RadioML2016.10A public-benchmark results
- Accuracy by SNR across main models
- Robustness drop across stress conditions
- Clean and worst-stress confusion matrices
- Runtime and parameter-count table
- Final manuscript, cover letter, title page, highlights, declarations
