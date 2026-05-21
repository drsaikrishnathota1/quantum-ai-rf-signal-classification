from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svm": "RBF-SVM",
    "simulated_quantum_feature_kernel_svm": "Simulated QFM-Kernel SVM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate synthetic RF analysis into paper assets.")
    parser.add_argument("--classical", type=Path, default=Path("results/synthetic_500/pilot_metrics.csv"))
    parser.add_argument("--classical-drops", type=Path, default=Path("results/synthetic_500/pilot_robustness_drop.csv"))
    parser.add_argument("--quantum", type=Path, default=Path("results/quantum_kernel_500/quantum_kernel_metrics.csv"))
    parser.add_argument("--quantum-drops", type=Path, default=Path("results/quantum_kernel_500/quantum_kernel_robustness_drop.csv"))
    parser.add_argument("--tables-out", type=Path, default=Path("manuscript_assets/tables"))
    parser.add_argument("--figures-out", type=Path, default=Path("manuscript_assets/figures"))
    parser.add_argument("--report-out", type=Path, default=Path("ANALYSIS_REPORT_SYNTHETIC_500.md"))
    return parser.parse_args()


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_clean_bar(clean: pd.DataFrame, out: Path) -> None:
    labels = clean["model_label"].tolist()
    values = clean["accuracy"].tolist()
    width, height = 1600, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(42, True)
    axis_font = font(25)
    label_font = font(22)
    draw.text((width // 2, 60), "Synthetic RF Clean Test Accuracy", fill="#111111", font=title_font, anchor="mm")
    x0, y0 = 150, 740
    plot_w, plot_h = 1350, 560
    draw.line((x0, y0 - plot_h, x0, y0), fill="#333333", width=3)
    draw.line((x0, y0, x0 + plot_w, y0), fill="#333333", width=3)
    for i in range(6):
        val = i / 5
        y = y0 - val * plot_h
        draw.line((x0 - 8, y, x0 + plot_w, y), fill="#E8E8E8", width=1)
        draw.text((x0 - 18, y), f"{val:.1f}", fill="#333333", font=axis_font, anchor="rm")
    bar_w = 210
    gap = (plot_w - len(values) * bar_w) / (len(values) + 1)
    colors = ["#2E74B5", "#70AD47", "#C55A11", "#8064A2"]
    for idx, (label, value) in enumerate(zip(labels, values)):
        x1 = x0 + gap + idx * (bar_w + gap)
        x2 = x1 + bar_w
        y1 = y0 - value * plot_h
        draw.rectangle((x1, y1, x2, y0), fill=colors[idx % len(colors)])
        draw.text(((x1 + x2) / 2, y1 - 24), f"{value:.3f}", fill="#111111", font=axis_font, anchor="mm")
        for line_idx, line in enumerate(label.split(" ")):
            draw.text(((x1 + x2) / 2, y0 + 35 + line_idx * 27), line, fill="#111111", font=label_font, anchor="mt")
    img.save(out, dpi=(300, 300))


def color_for_drop(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(90.0, value))
    intensity = int(255 - (value / 90.0) * 165)
    return (255, intensity, intensity)


def draw_drop_heatmap(drops: pd.DataFrame, out: Path) -> None:
    pivot = drops.pivot_table(index="condition", columns="model_label", values="accuracy_drop_pct")
    conditions = pivot.index.tolist()
    models = pivot.columns.tolist()
    cell_w, cell_h = 285, 70
    left, top = 300, 160
    width = left + cell_w * len(models) + 80
    height = top + cell_h * len(conditions) + 120
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(38, True)
    header_font = font(20, True)
    body_font = font(22)
    draw.text((width // 2, 58), "Accuracy Drop From Clean Under RF Stress Conditions (%)", fill="#111111", font=title_font, anchor="mm")
    for j, model in enumerate(models):
        x = left + j * cell_w
        draw.rectangle((x, top - cell_h, x + cell_w, top), fill="#F2F4F7", outline="#999999")
        chunks = model.split(" ")
        for k, chunk in enumerate(chunks[:3]):
            draw.text((x + cell_w / 2, top - 52 + k * 20), chunk, fill="#111111", font=header_font, anchor="mm")
    for i, condition in enumerate(conditions):
        y = top + i * cell_h
        draw.rectangle((0, y, left, y + cell_h), fill="#F8F8F8", outline="#CCCCCC")
        draw.text((left - 15, y + cell_h / 2), condition.replace("_", " "), fill="#111111", font=body_font, anchor="rm")
        for j, model in enumerate(models):
            x = left + j * cell_w
            value = float(pivot.loc[condition, model])
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color_for_drop(value), outline="#CCCCCC")
            draw.text((x + cell_w / 2, y + cell_h / 2), f"{value:.1f}", fill="#111111", font=body_font, anchor="mm")
    img.save(out, dpi=(300, 300))


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
    args.figures_out.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat(
        [pd.read_csv(args.classical), pd.read_csv(args.quantum)],
        ignore_index=True,
        sort=False,
    )
    drops = pd.concat(
        [pd.read_csv(args.classical_drops), pd.read_csv(args.quantum_drops)],
        ignore_index=True,
        sort=False,
    )
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    drops["model_label"] = drops["model"].map(MODEL_LABELS).fillna(drops["model"])

    clean = metrics[metrics["condition"] == "clean"].copy()
    clean = clean[["model", "model_label", "accuracy", "macro_f1"]].sort_values("accuracy", ascending=False)
    clean.to_csv(args.tables_out / "table_synthetic_clean_performance.csv", index=False)

    robustness = metrics[~metrics["condition"].isin(["heldout_clean", "clean"])].copy()
    robustness = robustness[["model", "model_label", "condition", "accuracy", "macro_f1"]]
    robustness.to_csv(args.tables_out / "table_synthetic_robustness_metrics.csv", index=False)

    drops = drops[["model", "model_label", "condition", "accuracy", "accuracy_drop_pct", "macro_f1", "macro_f1_drop_pct"]]
    drops.to_csv(args.tables_out / "table_synthetic_robustness_drop.csv", index=False)

    draw_clean_bar(clean, args.figures_out / "fig_synthetic_clean_accuracy.png")
    draw_drop_heatmap(drops, args.figures_out / "fig_synthetic_robustness_drop_heatmap.png")

    worst = drops.sort_values("accuracy_drop_pct", ascending=False).head(8)
    report = [
        "# Synthetic 500-Sample RF Analysis Report",
        "",
        "This report aggregates the corrected held-out synthetic RF evaluation.",
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
        "Random Forest provided the strongest clean accuracy among the classical pilot models, "
        "but all models showed substantial degradation under low-SNR, narrowband jamming, "
        "broadband jamming, and impulsive-noise conditions. The simulated quantum feature-map "
        "kernel baseline is currently weaker than the best classical model, which is important "
        "because the manuscript should not claim quantum advantage. Instead, the result supports "
        "a careful engineering framing: quantum-inspired feature maps can be evaluated as compact "
        "robustness-aware modules, but they must be compared against strong classical baselines.",
    ]
    args.report_out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote tables to {args.tables_out}")
    print(f"Wrote figures to {args.figures_out}")
    print(f"Wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
