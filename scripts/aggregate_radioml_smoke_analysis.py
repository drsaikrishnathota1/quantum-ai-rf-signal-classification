from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svm": "RBF-SVM",
    "simulated_quantum_feature_kernel_svm": "Simulated QFM-Kernel SVM",
    "iq_cnn": "Raw-IQ CNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RadioML2016.10A smoke-test results.")
    parser.add_argument("--classical", type=Path, default=Path("results/radioml2016_smoke_classical/pilot_metrics.csv"))
    parser.add_argument("--classical-drops", type=Path, default=Path("results/radioml2016_smoke_classical/pilot_robustness_drop.csv"))
    parser.add_argument("--classical-summary", type=Path, default=Path("results/radioml2016_smoke_classical/summary.json"))
    parser.add_argument("--cnn", type=Path, default=Path("results/radioml2016_smoke_cnn/cnn_metrics.csv"))
    parser.add_argument("--cnn-drops", type=Path, default=Path("results/radioml2016_smoke_cnn/cnn_robustness_drop.csv"))
    parser.add_argument(
        "--quantum",
        type=Path,
        default=Path("results/radioml2016_smoke_quantum_kernel/quantum_kernel_metrics.csv"),
    )
    parser.add_argument(
        "--quantum-drops",
        type=Path,
        default=Path("results/radioml2016_smoke_quantum_kernel/quantum_kernel_robustness_drop.csv"),
    )
    parser.add_argument("--tables-out", type=Path, default=Path("manuscript_assets/tables"))
    parser.add_argument("--report-out", type=Path, default=Path("PUBLIC_BENCHMARK_SMOKE_REPORT.md"))
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    args.tables_out.mkdir(parents=True, exist_ok=True)

    metric_frames = [pd.read_csv(args.classical), pd.read_csv(args.cnn), pd.read_csv(args.quantum)]
    drop_frames = [pd.read_csv(args.classical_drops), pd.read_csv(args.cnn_drops), pd.read_csv(args.quantum_drops)]
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    drops = pd.concat(drop_frames, ignore_index=True, sort=False)
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    drops["model_label"] = drops["model"].map(MODEL_LABELS).fillna(drops["model"])
    if "examples" not in metrics.columns:
        metrics["examples"] = pd.NA
    if args.classical_summary.exists():
        summary = json.loads(args.classical_summary.read_text(encoding="utf-8"))
        classical_examples = int(summary.get("test_examples", 0))
        classical_models = {"logistic_regression", "random_forest", "rbf_svm"}
        mask = metrics["model"].isin(classical_models) & metrics["examples"].isna()
        metrics.loc[mask, "examples"] = classical_examples

    clean = metrics[metrics["condition"] == "clean"].copy()
    clean = clean[["model", "model_label", "examples", "accuracy", "macro_f1"]].sort_values(
        "accuracy", ascending=False
    )
    clean.to_csv(args.tables_out / "table_radioml2016_smoke_clean_performance.csv", index=False)

    robustness = metrics[~metrics["condition"].isin(["heldout_clean", "clean"])].copy()
    robustness = robustness[["model", "model_label", "condition", "examples", "accuracy", "macro_f1"]]
    robustness.to_csv(args.tables_out / "table_radioml2016_smoke_robustness_metrics.csv", index=False)

    drops = drops[["model", "model_label", "condition", "accuracy", "accuracy_drop_pct", "macro_f1", "macro_f1_drop_pct"]]
    drops.to_csv(args.tables_out / "table_radioml2016_smoke_robustness_drop.csv", index=False)

    worst = drops.sort_values("accuracy_drop_pct", ascending=False).head(8)
    report = [
        "# RadioML2016.10A Public Benchmark Smoke Report",
        "",
        "This is a fast smoke test proving the public-benchmark workflow works end to end.",
        "It is not the final RadioML result table for the manuscript.",
        "",
        "## Dataset",
        "",
        "- Source file: `data/radioml/RML2016.10a_dict.pkl`",
        "- Smoke subset: 100 examples per modulation-SNR pair",
        "- Total smoke examples: 22,000",
        "- Modulation classes: 11",
        "- SNR values: -20 to 18 dB in 2 dB steps",
        "",
        "## Clean Performance",
        "",
        to_markdown_simple(clean),
        "",
        "## Worst Robustness Drops",
        "",
        to_markdown_simple(worst[["model_label", "condition", "accuracy", "accuracy_drop_pct", "macro_f1_drop_pct"]]),
        "",
        "## Interpretation",
        "",
        "The RadioML smoke run confirms that the public-benchmark ingestion, stress generation, "
        "classical baselines, CNN baseline, and simulated quantum feature-map kernel baseline "
        "all execute successfully. The current CNN result is intentionally under-trained "
        "because it uses only five local CPU epochs. Final manuscript claims should use a "
        "longer GPU run and report RadioML results separately from synthetic results.",
    ]
    args.report_out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote tables to {args.tables_out}")
    print(f"Wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
