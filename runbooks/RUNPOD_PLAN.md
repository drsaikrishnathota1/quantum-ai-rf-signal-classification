# RunPod Plan

## Recommended GPU

For the full paper-grade run:

- A100 40GB/80GB or H100 80GB
- 1 GPU is enough
- Estimated compute time: 6-12 hours across all model variants and robustness tests
- Estimated compute budget: USD 20-60 depending on GPU price and retries

## Storage

Use at least:

- Container disk: 80 GB
- Volume disk: 150-250 GB if using RadioML2018.01A

For the synthetic pilot and RadioML2016.10A only:

- 50 GB container disk
- 50-100 GB volume disk

## Environment

Use a PyTorch image with CUDA, then install:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm pyyaml joblib
pip install torch torchvision torchaudio
pip install qiskit qiskit-machine-learning
```

Optional:

```bash
pip install pennylane xgboost lightgbm
```

## First Commands

```bash
cd /workspace
git clone https://github.com/drsaikrishnathota1/quantum-ai-rf-signal-classification.git
cd quantum-ai-rf-signal-classification
pip install -r requirements.txt
python scripts/run_synthetic_pipeline.py --samples-per-class 500 --cnn-epochs 18
```

## Full Paper Workflow

1. Run synthetic pilot.
2. Scale synthetic dataset to 5k-10k samples per class.
3. Train classical baselines.
4. Train deep-learning baselines.
5. Run quantum-kernel baseline on compact features.
6. Download RadioML2016.10A and reproduce baseline comparison.
7. Run robustness matrix.
8. Export CSV tables and figures.
9. Generate manuscript package.

## Current One-Command Pipeline

For a stronger synthetic run on H100:

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
