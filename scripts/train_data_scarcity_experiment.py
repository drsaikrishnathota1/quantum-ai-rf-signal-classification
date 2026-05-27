"""
train_data_scarcity_experiment.py

RSC-Bench: Data-Scarcity Experiment
=====================================
Angle 1 experiment: Does the QFM-Kernel SVM hold up better than classical
models when training data is severely limited?

Sweeps training set sizes: [5, 10, 20, 50, 100, 200] samples per class.
For each size and each random seed, trains:
  - Logistic Regression
  - Random Forest
  - RBF-SVM
  - Simulated QFM-Kernel SVM

Outputs:
  - results/data_scarcity/scarcity_results.csv
  - manuscript_assets/tables/table_scarcity_accuracy_vs_trainsize.csv
  - manuscript_assets/figures/fig_scarcity_accuracy_vs_trainsize.png
  - manuscript_assets/figures/fig_scarcity_crossover_zoom.png

Usage:
  python scripts/train_data_scarcity_experiment.py \
      --data data/synthetic_iq_500.npz \
      --out results/data_scarcity \
      --seeds 5 \
      --qubits 5

The QFM-Kernel is the same simulated quantum feature-map kernel used in
train_quantum_inspired_kernel.py, so results are directly comparable.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Simulated Quantum Feature Map (matches existing pipeline logic)
# ---------------------------------------------------------------------------

def quantum_feature_map(X: np.ndarray, n_qubits: int = 5) -> np.ndarray:
    """
    Simulated quantum feature map using ZZFeatureMap-style angle encoding.

    Projects PCA-reduced features onto a 2*n_qubits dimensional space using
    first- and second-order Pauli-Z rotation terms. This is a classical
    simulation — no quantum hardware required.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features (should already be PCA-reduced to n_qubits dims).
    n_qubits : int
        Number of simulated qubits / input dimensions.

    Returns
    -------
    phi : ndarray of shape (n_samples, 2 * n_qubits)
        Quantum-inspired feature vectors.
    """
    n_samples = X.shape[0]
    # Clip to n_qubits features (PCA already done upstream)
    X_clipped = X[:, :n_qubits]

    # First-order terms: cos(pi * x_i)
    first_order = np.cos(np.pi * X_clipped)

    # Second-order terms: cos(pi * (pi - x_i)(pi - x_{i+1})) cyclic
    second_order = np.zeros((n_samples, n_qubits))
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        second_order[:, i] = np.cos(
            np.pi * (np.pi - X_clipped[:, i]) * (np.pi - X_clipped[:, j])
        )

    phi = np.hstack([first_order, second_order])
    return phi


def build_qfm_pipeline(n_qubits: int = 5) -> Pipeline:
    """
    Build a pipeline: StandardScaler -> PCA -> QFM -> RBF-SVM.
    Matches the existing train_quantum_inspired_kernel.py approach.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import FunctionTransformer

    pca = PCA(n_components=n_qubits)

    def _qfm(X):
        return quantum_feature_map(X, n_qubits=n_qubits)

    qfm_transformer = FunctionTransformer(_qfm)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", pca),
        ("qfm", qfm_transformer),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])
    return pipe


# ---------------------------------------------------------------------------
# Classical model builders
# ---------------------------------------------------------------------------

def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0)),
    ])


def build_rf():
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])


def build_rbf_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str):
    data = np.load(path, allow_pickle=True)
    X = data["X_clean"]           # shape (N, 2, 128) or (N, 256)
    y = data["y"]
    # Flatten IQ to 1D if needed
    if X.ndim == 3:
        X = X.reshape(len(X), -1)
    return X, y


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

MODELS = {
    "Logistic Regression": build_lr,
    "Random Forest":       build_rf,
    "RBF-SVM":             build_rbf_svm,
    "QFM-Kernel SVM":      None,   # built per-run with n_qubits
}

TRAIN_SIZES = [5, 10, 20, 50, 100, 200]   # samples per class
TEST_PER_CLASS = 60                         # fixed held-out pool per class


def run_experiment(X, y, train_sizes, seeds, n_qubits, out_dir):
    classes = np.unique(y)
    n_classes = len(classes)
    records = []

    # Build a fixed large held-out pool (same across all train sizes)
    rng_master = np.random.RandomState(42)
    test_idx = []
    for c in classes:
        cidx = np.where(y == c)[0]
        rng_master.shuffle(cidx)
        test_idx.extend(cidx[:TEST_PER_CLASS].tolist())
    test_idx = np.array(test_idx)
    X_test, y_test = X[test_idx], y[test_idx]

    # All remaining indices available for training
    all_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    total = len(MODELS) * len(train_sizes) * seeds
    done = 0

    for train_size in train_sizes:
        for seed in range(seeds):
            rng = np.random.RandomState(seed)

            # Sample train_size examples per class from available pool
            train_idx = []
            for c in classes:
                cidx = np.intersect1d(all_idx, np.where(y == c)[0])
                if len(cidx) < train_size:
                    # If not enough samples, use all available (warn)
                    chosen = cidx
                else:
                    perm = rng.permutation(len(cidx))
                    chosen = cidx[perm[:train_size]]
                train_idx.extend(chosen.tolist())

            train_idx = np.array(train_idx)
            X_train, y_train = X[train_idx], y[train_idx]

            for model_name, builder in MODELS.items():
                if builder is None:
                    model = build_qfm_pipeline(n_qubits=n_qubits)
                else:
                    model = builder()

                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                except Exception as e:
                    print(f"  WARNING: {model_name} train_size={train_size} "
                          f"seed={seed} failed: {e}")
                    acc, f1 = np.nan, np.nan

                records.append({
                    "model": model_name,
                    "train_size_per_class": train_size,
                    "total_train_samples": train_size * n_classes,
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": f1,
                })

                done += 1
                print(f"  [{done}/{total}] {model_name:25s} | "
                      f"n={train_size:4d}/class | seed={seed} | "
                      f"acc={acc:.4f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation and summary table
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "train_size_per_class"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            std_f1=("macro_f1", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return grouped


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "Logistic Regression": "#4e79a7",
    "Random Forest":       "#f28e2b",
    "RBF-SVM":             "#e15759",
    "QFM-Kernel SVM":      "#59a14f",
}

MODEL_MARKERS = {
    "Logistic Regression": "o",
    "Random Forest":       "s",
    "RBF-SVM":             "^",
    "QFM-Kernel SVM":      "D",
}


def plot_scarcity_curve(agg_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model, grp in agg_df.groupby("model"):
        grp = grp.sort_values("train_size_per_class")
        ax.errorbar(
            grp["train_size_per_class"],
            grp["mean_accuracy"],
            yerr=grp["std_accuracy"],
            label=model,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=2,
            markersize=6,
            capsize=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training samples per class (log scale)", fontsize=12)
    ax.set_ylabel("Mean Accuracy ± 1 std", fontsize=12)
    ax.set_title("Accuracy vs. Training Set Size\n"
                 "(Quantum-Inspired vs. Classical Methods)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(TRAIN_SIZES)
    ax.set_xticklabels([str(s) for s in TRAIN_SIZES])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_crossover_zoom(agg_df: pd.DataFrame, out_path: str):
    """Zoom into the low-data regime (5–50 samples/class)."""
    zoom_df = agg_df[agg_df["train_size_per_class"] <= 50]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, grp in zoom_df.groupby("model"):
        grp = grp.sort_values("train_size_per_class")
        ax.errorbar(
            grp["train_size_per_class"],
            grp["mean_accuracy"],
            yerr=grp["std_accuracy"],
            label=model,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=2.5,
            markersize=7,
            capsize=5,
        )

    ax.set_xlabel("Training samples per class", fontsize=12)
    ax.set_ylabel("Mean Accuracy ± 1 std", fontsize=12)
    ax.set_title("Low-Data Regime: Does QFM-Kernel Close the Gap?", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([5, 10, 20, 50])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Narrative interpretation helper
# ---------------------------------------------------------------------------

def interpret_results(agg_df: pd.DataFrame) -> dict:
    """
    Automatically detect whether a crossover exists:
    i.e., QFM-Kernel ranks better (relative to classical) at small train sizes.
    Returns a dict with key findings for the manuscript narrative.
    """
    findings = {}

    qfm = agg_df[agg_df["model"] == "QFM-Kernel SVM"].set_index("train_size_per_class")
    rf  = agg_df[agg_df["model"] == "Random Forest"].set_index("train_size_per_class")

    if qfm.empty or rf.empty:
        return {"crossover_detected": False}

    # At each train size, rank QFM vs RF
    sizes = sorted(qfm.index.tolist())
    qfm_beats_rf = {}
    for s in sizes:
        if s in rf.index:
            qfm_beats_rf[s] = qfm.loc[s, "mean_accuracy"] >= rf.loc[s, "mean_accuracy"]

    crossover_sizes = [s for s, wins in qfm_beats_rf.items() if wins]
    findings["crossover_detected"] = len(crossover_sizes) > 0
    findings["qfm_competitive_at_sizes"] = crossover_sizes
    findings["qfm_best_accuracy"] = float(qfm["mean_accuracy"].max())
    findings["qfm_best_accuracy_size"] = int(qfm["mean_accuracy"].idxmax())
    findings["rf_best_accuracy"] = float(rf["mean_accuracy"].max())

    # Gap at smallest size
    s_min = sizes[0]
    if s_min in rf.index:
        gap_small = float(rf.loc[s_min, "mean_accuracy"] - qfm.loc[s_min, "mean_accuracy"])
        findings["gap_at_smallest_size"] = gap_small
        findings["smallest_train_size"] = s_min

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RSC-Bench data-scarcity experiment: QFM-Kernel vs classical"
    )
    parser.add_argument("--data",   default="data/synthetic_iq_500.npz",
                        help="Path to synthetic IQ .npz dataset")
    parser.add_argument("--out",    default="results/data_scarcity",
                        help="Output directory for results")
    parser.add_argument("--seeds",  type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--qubits", type=int, default=5,
                        help="Number of simulated qubits (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("manuscript_assets/tables", exist_ok=True)
    os.makedirs("manuscript_assets/figures", exist_ok=True)

    print(f"\n{'='*60}")
    print("RSC-Bench: Data-Scarcity Experiment")
    print(f"{'='*60}")
    print(f"  Data:    {args.data}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Qubits:  {args.qubits}")
    print(f"  Train sizes/class: {TRAIN_SIZES}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading dataset...")
    X, y = load_data(args.data)
    print(f"  X shape: {X.shape}, classes: {np.unique(y)}\n")

    # Run experiment
    print("Running scarcity sweep...\n")
    raw_df = run_experiment(X, y, TRAIN_SIZES, args.seeds, args.qubits, args.out)

    # Save raw results
    raw_path = os.path.join(args.out, "scarcity_raw_results.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  Saved raw results: {raw_path}")

    # Aggregate
    agg_df = aggregate(raw_df)
    agg_path = os.path.join(args.out, "scarcity_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)

    # Manuscript table
    ms_table_path = "manuscript_assets/tables/table_scarcity_accuracy_vs_trainsize.csv"
    # Pivot for clean manuscript table
    pivot = agg_df.pivot_table(
        index="model",
        columns="train_size_per_class",
        values="mean_accuracy",
    ).round(4)
    pivot.to_csv(ms_table_path)
    print(f"  Saved manuscript table: {ms_table_path}")

    # Figures
    print("\nGenerating figures...")
    plot_scarcity_curve(
        agg_df,
        "manuscript_assets/figures/fig_scarcity_accuracy_vs_trainsize.png"
    )
    plot_crossover_zoom(
        agg_df,
        "manuscript_assets/figures/fig_scarcity_crossover_zoom.png"
    )

    # Interpret findings
    findings = interpret_results(agg_df)
    findings_path = os.path.join(args.out, "scarcity_findings.json")
    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2)

    # Print narrative summary
    print(f"\n{'='*60}")
    print("FINDINGS SUMMARY")
    print(f"{'='*60}")
    if findings.get("crossover_detected"):
        print(f"  ✅ CROSSOVER DETECTED at train sizes: "
              f"{findings['qfm_competitive_at_sizes']} samples/class")
        print(f"     → QFM-Kernel SVM is competitive with Random Forest "
              f"at very low data volumes.")
        print(f"     → This is your publishable quantum finding.")
    else:
        print(f"  ⚠️  No crossover detected.")
        print(f"     Gap at {findings.get('smallest_train_size')} samples/class: "
              f"{findings.get('gap_at_smallest_size', 'N/A'):.4f}")
        print(f"     → Reframe quantum component as an honest baseline comparison.")
        print(f"     → The finding is: QFM-Kernel underperforms across all regimes")
        print(f"       on this dataset, which is itself a valid scientific result.")

    print(f"\n  QFM best accuracy: {findings.get('qfm_best_accuracy', 'N/A'):.4f} "
          f"at {findings.get('qfm_best_accuracy_size', 'N/A')} samples/class")
    print(f"  RF  best accuracy: {findings.get('rf_best_accuracy', 'N/A'):.4f}")
    print(f"\n  Full findings: {findings_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
