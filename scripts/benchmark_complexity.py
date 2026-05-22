"""Benchmark computational complexity and inference latency for all RF classifiers.

Produces a ``complexity_metrics.csv`` table with parameter counts, training
times, and per-sample inference latency suitable for inclusion in the manuscript
as a computational-complexity comparison table.

Usage
-----
    .venv/bin/python scripts/benchmark_complexity.py \
        --data data/synthetic_iq_500.npz \
        --out manuscript_assets/tables/complexity_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from train_pilot_classifiers import extract_features
from train_quantum_inspired_kernel import quantum_feature_state, quantum_kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model complexity and inference latency."
    )
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_iq_500.npz"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manuscript_assets/tables/complexity_metrics.csv"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--latency-runs", type=int, default=1000)
    parser.add_argument("--cnn-epochs", type=int, default=18)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--quantum-qubits", type=int, default=5)
    return parser.parse_args()


def count_torch_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency_sklearn(model, x_single: np.ndarray, n_runs: int) -> float:
    """Return average inference latency in milliseconds for a single sample."""
    x = x_single.reshape(1, -1)
    # warm up
    for _ in range(10):
        model.predict(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times))


def measure_latency_torch(model: torch.nn.Module, x_single: np.ndarray, n_runs: int) -> float:
    """Return average inference latency in milliseconds for a single sample."""
    device = next(model.parameters()).device
    x = torch.tensor(x_single, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        # warm up
        for _ in range(10):
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.mean(times))


def measure_latency_quantum_kernel(
    sample_raw: np.ndarray,
    feature_scaler: StandardScaler,
    pca: PCA,
    angle_scaler: MinMaxScaler,
    train_states: np.ndarray,
    clf: SVC,
    entangle_strength: float,
    n_runs: int,
) -> float:
    """Return average inference latency in ms for the quantum kernel pipeline."""
    # warm up
    for _ in range(5):
        feat = feature_scaler.transform(extract_features(sample_raw.reshape(1, *sample_raw.shape)))
        angles = angle_scaler.transform(pca.transform(feat))
        state = quantum_feature_state(angles, entangle_strength)
        k = quantum_kernel(state, train_states)
        clf.predict(k)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        feat = feature_scaler.transform(extract_features(sample_raw.reshape(1, *sample_raw.shape)))
        angles = angle_scaler.transform(pca.transform(feat))
        state = quantum_feature_state(angles, entangle_strength)
        k = quantum_kernel(state, train_states)
        clf.predict(k)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    modulations = [str(v) for v in data["modulations"]]
    num_classes = len(modulations)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=args.seed, stratify=y
    )
    x_train_raw, y_train = x[train_idx], y[train_idx]
    x_test_raw, y_test = x[test_idx], y[test_idx]

    x_train_feat = extract_features(x_train_raw)
    x_test_feat = extract_features(x_test_raw)

    records = []

    # ---- 1. Logistic Regression ---- #
    lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, C=2.0))])
    t0 = time.perf_counter()
    lr.fit(x_train_feat, y_train)
    lr_train_time = time.perf_counter() - t0
    lr_params = int(lr["clf"].coef_.size + lr["clf"].intercept_.size)
    lr_latency = measure_latency_sklearn(lr, x_test_feat[0], args.latency_runs)
    records.append({
        "model": "Logistic Regression",
        "parameters": lr_params,
        "training_time_s": round(lr_train_time, 3),
        "inference_latency_ms": round(lr_latency, 4),
    })

    # ---- 2. Random Forest ---- #
    rf = RandomForestClassifier(n_estimators=250, random_state=args.seed, min_samples_leaf=2)
    t0 = time.perf_counter()
    rf.fit(x_train_feat, y_train)
    rf_train_time = time.perf_counter() - t0
    rf_params = sum(tree.tree_.node_count for tree in rf.estimators_)
    rf_latency = measure_latency_sklearn(rf, x_test_feat[0], args.latency_runs)
    records.append({
        "model": "Random Forest",
        "parameters": rf_params,
        "training_time_s": round(rf_train_time, 3),
        "inference_latency_ms": round(rf_latency, 4),
    })

    # ---- 3. RBF-SVM ---- #
    svm = Pipeline([("scale", StandardScaler()), ("clf", SVC(C=8.0, gamma="scale"))])
    t0 = time.perf_counter()
    svm.fit(x_train_feat, y_train)
    svm_train_time = time.perf_counter() - t0
    svm_params = int(svm["clf"].support_vectors_.size) + int(svm["clf"].dual_coef_.size)
    svm_latency = measure_latency_sklearn(svm, x_test_feat[0], args.latency_runs)
    records.append({
        "model": "RBF-SVM",
        "parameters": svm_params,
        "training_time_s": round(svm_train_time, 3),
        "inference_latency_ms": round(svm_latency, 4),
    })

    # ---- 4. Raw-IQ CNN ---- #
    from train_cnn_iq_baseline import IQCNN
    device = torch.device("cpu")
    cnn = IQCNN(num_classes=num_classes).to(device)
    cnn_params = count_torch_params(cnn)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    xt = torch.tensor(x_train_raw, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    train_ds = torch.utils.data.TensorDataset(xt, yt)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.cnn_batch_size, shuffle=True)

    t0 = time.perf_counter()
    for epoch in range(args.cnn_epochs):
        cnn.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(cnn(xb), yb)
            loss.backward()
            optimizer.step()
    cnn_train_time = time.perf_counter() - t0
    cnn_latency = measure_latency_torch(cnn, x_test_raw[0], args.latency_runs)
    records.append({
        "model": "Raw-IQ CNN",
        "parameters": cnn_params,
        "training_time_s": round(cnn_train_time, 3),
        "inference_latency_ms": round(cnn_latency, 4),
    })

    # ---- 5. Simulated Quantum-Feature-Map Kernel SVM ---- #
    feature_scaler = StandardScaler()
    pca = PCA(n_components=args.quantum_qubits, random_state=args.seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    train_features = feature_scaler.fit_transform(x_train_feat)
    train_pca = pca.fit_transform(train_features)
    train_angles = angle_scaler.fit_transform(train_pca)

    t0 = time.perf_counter()
    train_states = quantum_feature_state(train_angles, 0.35)
    k_train = quantum_kernel(train_states, train_states)
    qsvm = SVC(kernel="precomputed", C=4.0)
    qsvm.fit(k_train, y_train)
    qsvm_train_time = time.perf_counter() - t0
    qsvm_params = int(qsvm.support_vectors_.size) + int(qsvm.dual_coef_.size)
    qsvm_latency = measure_latency_quantum_kernel(
        x_test_raw[0], feature_scaler, pca, angle_scaler,
        train_states, qsvm, 0.35, args.latency_runs,
    )
    records.append({
        "model": "Simulated QFM-Kernel SVM",
        "parameters": qsvm_params,
        "training_time_s": round(qsvm_train_time, 3),
        "inference_latency_ms": round(qsvm_latency, 4),
    })

    # ---- 6. Classical PCA-RBF SVM ---- #
    classical_pca = Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", C=4.0, gamma="scale"))])
    t0 = time.perf_counter()
    classical_pca.fit(train_pca, y_train)
    cpca_train_time = time.perf_counter() - t0
    cpca_params = int(classical_pca["clf"].support_vectors_.size) + int(classical_pca["clf"].dual_coef_.size)
    cpca_latency = measure_latency_sklearn(classical_pca, pca.transform(feature_scaler.transform(x_test_feat[0:1]))[0], args.latency_runs)
    records.append({
        "model": "Classical PCA-RBF SVM",
        "parameters": cpca_params,
        "training_time_s": round(cpca_train_time, 3),
        "inference_latency_ms": round(cpca_latency, 4),
    })

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved complexity metrics to {args.out}")


if __name__ == "__main__":
    main()
