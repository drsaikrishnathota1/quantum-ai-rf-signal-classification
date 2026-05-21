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
git clone <REPO_URL_IF_CREATED> quantum_ai_rf_signal_classification
cd quantum_ai_rf_signal_classification
pip install -r requirements.txt
python scripts/generate_synthetic_iq_dataset.py --samples-per-class 1000 --out data/synthetic_iq_1k.npz
python scripts/train_pilot_classifiers.py --data data/synthetic_iq_1k.npz --out results/pilot_1k
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

