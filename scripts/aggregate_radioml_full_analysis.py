from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svm": "RBF-SVM",
    "simulated_quantum_feature_kernel_svm": "Simulated QFM-Kernel SVM",
    "iq_cnn": "Raw-IQ CNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full RadioML2016.10A benchmark results.")
    parser.add_argument("--classical", type=Path, default=Path("results/radioml2016_classical/pilot_metrics.csv"))
    parser.add_argument(
        "--classical-drops",
        type=Path,
        default=Path("results/radioml2016_classical/pilot_robustness_drop.csv"),
    )
    parser.add_argument(
        "--classical-summary",
        type=Path,
        default=Path("results/radioml2016_classical/summary.json"),
    )
    parser.add_argument("--cnn", type=Path, default=Path("results/radioml2016_cnn/cnn_metrics.csv"))
    parser.add_argument("--cnn-drops", type=Path, default=Path("results/radioml2016_cnn/cnn_robustness_drop.csv"))
    parser.add_argument("--cnn-summary", type=Path, default=Path("results/radioml2016_cnn/summary.json"))
    parser.add_argument(
        "--quantum",
        type=Path,
        default=Path("results/radioml2016_quantum_kernel/quantum_kernel_metrics.csv"),
    )
    parser.add_argument(
        "--quantum-drops",
        type=Path,
        default=Path("results/radioml2016_quantum_kernel/quantum_kernel_robustness_drop.csv"),
    )
    parser.add_argument(
        "--quantum-summary",
        type=Path,
        default=Path("results/radioml2016_quantum_kernel/summary.json"),
    )
    parser.add_argument("--tables-out", type=Path, default=Path("manuscript_assets/tables"))
    parser.add_argument("--figures-out", type=Path, default=Path("manuscript_assets/figures"))
    parser.add_argument("--report-out", type=Path, default=Path("RADIOML2016_FULL_GPU_REPORT.md"))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_markdown_simple(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_clean_accuracy_figure(clean: pd.DataFrame, out: Path) -> None:
    plot_df = clean.copy().sort_values("accuracy", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(plot_df["model_label"], plot_df["accuracy"], color="#2868a8")
    plt.xlabel("Held-out clean accuracy")
    plt.xlim(0, max(0.6, float(plot_df["accuracy"].max()) + 0.05))
    plt.title("RadioML2016.10A clean-set classification performance")
    plt.tight_layout()
    plt.savefig(out / "fig_radioml2016_clean_accuracy.png", dpi=300)
    plt.savefig(out / "fig_radioml2016_clean_accuracy.pdf")
    plt.close()


def save_robustness_drop_figure(drops: pd.DataFrame, out: Path) -> None:
    plot_df = drops.copy()
    best_clean_model = (
        plot_df.groupby("model_label")["accuracy_drop_pct"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    pivot = plot_df.pivot_table(index="condition", columns="model_label", values="accuracy_drop_pct", aggfunc="mean")
    pivot = pivot.reindex(columns=best_clean_model)
    pivot.plot(kind="bar", figsize=(10, 5.5))
    plt.ylabel("Accuracy drop from clean (%)")
    plt.xlabel("Stress condition")
    plt.title("RadioML2016.10A robustness degradation by model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_radioml2016_robustness_drop.png", dpi=300)
    plt.savefig(out / "fig_radioml2016_robustness_drop.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    args.tables_out.mkdir(parents=True, exist_ok=True)
    args.figures_out.mkdir(parents=True, exist_ok=True)

    metric_frames = [pd.read_csv(args.classical), pd.read_csv(args.cnn), pd.read_csv(args.quantum)]
    drop_frames = [pd.read_csv(args.classical_drops), pd.read_csv(args.cnn_drops), pd.read_csv(args.quantum_drops)]
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    drops = pd.concat(drop_frames, ignore_index=True, sort=False)
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    drops["model_label"] = drops["model"].map(MODEL_LABELS).fillna(drops["model"])

    clean = metrics[metrics["condition"] == "clean"].copy()
    clean = clean[["model", "model_label", "examples", "accuracy", "macro_f1"]].sort_values(
        "accuracy",
        ascending=False,
    )
    clean.to_csv(args.tables_out / "table_radioml2016_full_clean_performance.csv", index=False)

    robustness = metrics[~metrics["condition"].isin(["heldout_clean", "clean"])].copy()
    robustness = robustness[["model", "model_label", "condition", "examples", "accuracy", "macro_f1"]]
    robustness.to_csv(args.tables_out / "table_radioml2016_full_robustness_metrics.csv", index=False)

    drops = drops[
        ["model", "model_label", "condition", "accuracy", "accuracy_drop_pct", "macro_f1", "macro_f1_drop_pct"]
    ].sort_values(["model_label", "accuracy_drop_pct"], ascending=[True, False])
    drops.to_csv(args.tables_out / "table_radioml2016_full_robustness_drop.csv", index=False)

    method_config = pd.DataFrame(
        [
            {
                "method": "Classical baselines",
                "models": "Logistic Regression; Random Forest; RBF-SVM",
                "train_examples": read_json(args.classical_summary).get("train_examples", "30000 cap"),
                "test_examples": read_json(args.classical_summary).get("test_examples", "10000 cap"),
            },
            {
                "method": "Raw-IQ CNN",
                "models": "Compact 1D convolutional neural network",
                "train_examples": read_json(args.cnn_summary).get("train_examples", "80000 cap"),
                "test_examples": read_json(args.cnn_summary).get("test_examples", "20000 cap"),
            },
            {
                "method": "Simulated QFM-Kernel SVM",
                "models": "Five-qubit simulated feature-map kernel with SVM",
                "train_examples": "120 per class cap",
                "test_examples": "80 per class cap",
            },
        ]
    )
    method_config.to_csv(args.tables_out / "table_radioml2016_full_method_config.csv", index=False)

    save_clean_accuracy_figure(clean, args.figures_out)
    save_robustness_drop_figure(drops, args.figures_out)

    worst = drops.sort_values("accuracy_drop_pct", ascending=False).head(10)
    best_clean = clean.iloc[0]
    report = [
        "# RadioML2016.10A Full GPU Benchmark Report",
        "",
        "This report aggregates the full public-benchmark workflow executed on RadioML2016.10A.",
        "It should be used as the primary manuscript evidence for the public RF benchmark.",
        "",
        "## Dataset",
        "",
        "- Dataset: RadioML2016.10A",
        "- Total examples in converted clean NPZ: 220,000",
        "- Modulation classes: 11",
        "- SNR values: -20 to 18 dB in 2 dB steps",
        "- Stress conditions evaluated: clean, low_snr, narrowband_jam, broadband_jam, frequency_offset, multipath, impulsive_noise",
        "",
        "## Execution Notes",
        "",
        "- Classical models used capped stratified training and test subsets for tractable baseline comparison.",
        "- Raw-IQ CNN used CUDA GPU training with best-epoch restoration.",
        "- The quantum-inspired result is a simulated feature-map kernel baseline; no quantum advantage is claimed.",
        "",
        "## Method Configuration",
        "",
        to_markdown_simple(method_config),
        "",
        "## Clean Performance",
        "",
        to_markdown_simple(clean),
        "",
        "## Best Clean Result",
        "",
        f"- Best clean model: {best_clean['model_label']}",
        f"- Clean accuracy: {best_clean['accuracy']:.4f}",
        f"- Clean macro-F1: {best_clean['macro_f1']:.4f}",
        "",
        "## Worst Robustness Drops",
        "",
        to_markdown_simple(worst[["model_label", "condition", "accuracy", "accuracy_drop_pct", "macro_f1_drop_pct"]]),
        "",
        "## Generated Tables",
        "",
        "- `manuscript_assets/tables/table_radioml2016_full_clean_performance.csv`",
        "- `manuscript_assets/tables/table_radioml2016_full_robustness_metrics.csv`",
        "- `manuscript_assets/tables/table_radioml2016_full_robustness_drop.csv`",
        "- `manuscript_assets/tables/table_radioml2016_full_method_config.csv`",
        "",
        "## Generated Figures",
        "",
        "- `manuscript_assets/figures/fig_radioml2016_clean_accuracy.png`",
        "- `manuscript_assets/figures/fig_radioml2016_robustness_drop.png`",
        "",
        "## Manuscript Interpretation",
        "",
        "The full public benchmark supports a cautious engineering conclusion: the compact Raw-IQ CNN "
        "provided the strongest clean-set performance among the tested methods, while stress tests show "
        "large degradation under contested-spectrum perturbations. The simulated quantum feature-map "
        "kernel is useful as a transparent quantum-inspired comparator, but the observed performance "
        "does not support a quantum-advantage claim.",
    ]
    args.report_out.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote tables to {args.tables_out}")
    print(f"Wrote figures to {args.figures_out}")
    print(f"Wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
