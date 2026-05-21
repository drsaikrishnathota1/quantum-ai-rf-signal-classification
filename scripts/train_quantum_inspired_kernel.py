from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from train_pilot_classifiers import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact simulated quantum-feature-map kernel SVM on RF features."
    )
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_iq_500.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/quantum_kernel_500"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--max-train-per-class", type=int, default=90)
    parser.add_argument("--max-test-per-class", type=int, default=60)
    parser.add_argument("--entangle-strength", type=float, default=0.35)
    return parser.parse_args()


def stratified_limit_indices(
    indices: np.ndarray,
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = []
    for cls in np.unique(y):
        cls_idx = indices[y[indices] == cls]
        take = min(per_class, len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    idx = np.asarray(idx)
    rng.shuffle(idx)
    return idx


def quantum_feature_state(theta: np.ndarray, entangle_strength: float) -> np.ndarray:
    """Build a small complex statevector from angle features.

    This is a simulated quantum-inspired feature map. It uses tensor-product
    single-qubit angle features plus pairwise phase terms. The paper should
    describe it as a compact NISQ-style feature-map simulator, not hardware
    quantum advantage.
    """
    n_samples, n_qubits = theta.shape
    dim = 2**n_qubits
    states = np.empty((n_samples, dim), dtype=np.complex64)
    bit_patterns = np.array(
        [[(basis >> q) & 1 for q in range(n_qubits)] for basis in range(dim)],
        dtype=np.float32,
    )
    for i in range(n_samples):
        angles = theta[i]
        amps = np.ones(dim, dtype=np.float32)
        for q in range(n_qubits):
            amps *= np.where(
                bit_patterns[:, q] > 0,
                np.sin(angles[q] / 2.0),
                np.cos(angles[q] / 2.0),
            )
        phase = np.zeros(dim, dtype=np.float32)
        for a in range(n_qubits):
            for b in range(a + 1, n_qubits):
                phase += (
                    entangle_strength
                    * bit_patterns[:, a]
                    * bit_patterns[:, b]
                    * angles[a]
                    * angles[b]
                    / np.pi
                )
        states[i] = amps * np.exp(1j * phase)
    norms = np.linalg.norm(states, axis=1, keepdims=True) + 1e-12
    return states / norms


def quantum_kernel(a_states: np.ndarray, b_states: np.ndarray) -> np.ndarray:
    overlaps = a_states @ b_states.conj().T
    return np.abs(overlaps) ** 2


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.data, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    modulations = [str(v) for v in data["modulations"]]
    stress_conditions = [str(v) for v in data["stress_conditions"]]

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )
    train_idx = stratified_limit_indices(train_idx, y, args.max_train_per_class, rng)
    test_idx = stratified_limit_indices(test_idx, y, args.max_test_per_class, rng)
    x_train_raw, y_train = x[train_idx], y[train_idx]
    x_test_raw, y_test = x[test_idx], y[test_idx]

    feature_scaler = StandardScaler()
    pca = PCA(n_components=args.qubits, random_state=args.seed)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    train_features = feature_scaler.fit_transform(extract_features(x_train_raw))
    train_angles = angle_scaler.fit_transform(pca.fit_transform(train_features))
    train_states = quantum_feature_state(train_angles, args.entangle_strength)
    k_train = quantum_kernel(train_states, train_states)

    clf = SVC(kernel="precomputed", C=4.0)
    clf.fit(k_train, y_train)

    def transform_states(x_eval: np.ndarray) -> np.ndarray:
        features = feature_scaler.transform(extract_features(x_eval))
        angles = angle_scaler.transform(pca.transform(features))
        return quantum_feature_state(angles, args.entangle_strength)

    def evaluate(condition: str, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float | str | int]:
        states = transform_states(x_eval)
        k_eval = quantum_kernel(states, train_states)
        pred = clf.predict(k_eval)
        return {
            "model": "simulated_quantum_feature_kernel_svm",
            "condition": condition,
            "examples": int(len(y_eval)),
            "accuracy": float(accuracy_score(y_eval, pred)),
            "macro_f1": float(f1_score(y_eval, pred, average="macro")),
        }

    records = [evaluate("heldout_clean", x_test_raw, y_test)]

    for condition in stress_conditions:
        xs = data[f"X_{condition}"][test_idx]
        ys = data[f"y_{condition}"][test_idx]
        records.append(evaluate(condition, xs, ys))

    df = pd.DataFrame(records)
    clean = df[df["condition"] == "clean"].iloc[0]
    drop_rows = []
    for _, row in df.iterrows():
        if row["condition"] in {"heldout_clean", "clean"}:
            continue
        drop_rows.append(
            {
                "model": row["model"],
                "condition": row["condition"],
                "accuracy": row["accuracy"],
                "accuracy_drop_pct": float((clean["accuracy"] - row["accuracy"]) / clean["accuracy"] * 100),
                "macro_f1": row["macro_f1"],
                "macro_f1_drop_pct": float((clean["macro_f1"] - row["macro_f1"]) / clean["macro_f1"] * 100),
            }
        )

    df.to_csv(args.out / "quantum_kernel_metrics.csv", index=False)
    pd.DataFrame(drop_rows).to_csv(args.out / "quantum_kernel_robustness_drop.csv", index=False)
    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "pca": pca,
            "angle_scaler": angle_scaler,
            "clf": clf,
            "train_states": train_states,
            "train_labels": y_train,
        },
        args.out / "quantum_kernel_model.joblib",
    )
    summary = {
        "dataset": str(args.data),
        "modulations": modulations,
        "stress_conditions": stress_conditions,
        "qubits": args.qubits,
        "train_examples": int(len(y_train)),
        "test_examples_per_condition": int(args.max_test_per_class * len(np.unique(y))),
        "entangle_strength": args.entangle_strength,
        "clean_accuracy": float(clean["accuracy"]),
        "clean_macro_f1": float(clean["macro_f1"]),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.round(4).to_string(index=False))
    print(f"Saved quantum-inspired kernel outputs to {args.out}")


if __name__ == "__main__":
    main()
